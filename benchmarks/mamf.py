import datetime
import argparse
import numpy as np
import os
import sys
import time
import torch
import optuna
from optuna.storages import RDBStorage
import warnings
from pathlib import Path
from rich import print as rprint
from utils.shared_state import TERMINATE_REQUESTED, should_terminate
from utils.db_utils import save_benchmark_result, update_benchmark_summary

from utils.arch import get_accelerator_arch

warnings.filterwarnings("ignore", message="set_metric_names is experimental*")

# Get architecture singleton
arch = get_accelerator_arch()

def add_benchmark_args(parser):
    """Add MAMF-specific arguments to the ArgumentParser"""
    parser.add_argument(
        "--m_range",
        nargs=3,
        type=int,
        default=[1024, 20480, 64],
        help="The first dimension of the GEMM, [start,stop,step]",
    )
    parser.add_argument(
        "--n_range",
        nargs=3,
        type=int,
        default=[1024, 20480, 64],
        help="The shared dimension of the GEMM, [start,stop,step]",
    )
    parser.add_argument(
        "--k_range",
        nargs=3,
        type=int,
        default=[1024, 20480, 64],
        help="The last dimension of the GEMM, [start,stop,step]",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="The number of iterations used to benchmark each GEMM",
    )
    parser.add_argument(
        "--num_warmup_iterations",
        type=int,
        default=50,
        help="The number of warmup iterations",
    )
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=0,
        help="The cuda device to run the benchmark on",
    )
    parser.add_argument(
        "--n_trials", 
        type=int, 
        default=1000, 
        help="Number of trials for Optuna"
    )
    parser.add_argument(
        "--study_name", 
        type=str, 
        default="mamf_study", 
        help="Name of the Optuna study"
    )
    parser.add_argument(
        "--db_file",
        type=str,
        default="mamf.db",
        help="Database file to store Optuna results",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default=None,
        help="Path to summary file for updating benchmark results (default: None)"
    )

def benchmark_mm(m, n, k, dtype, device, num_iterations, num_warmup_iterations):
    """Benchmark a single matrix multiplication operation"""
    start = arch.event(enable_timing=True)
    end = arch.event(enable_timing=True)

    A = torch.randn(m, n, dtype=dtype, device=device)
    B = torch.randn(n, k, dtype=dtype, device=device)
    C = torch.empty(m, k, dtype=dtype, device=device)

    times = np.zeros(num_iterations + num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        arch.clear_cache()
        with torch.no_grad():
            start.record()
            torch.mm(A, B, out=C)
            end.record()
        arch.synchronize()
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.median(times) / 1000  # want the median
    tflops = (2 * m * n * k) / (elapsed_time * 10**12)
    return tflops

def objective(trial, args, dtype, device):
    """Objective function for Optuna optimization"""
    M = trial.suggest_int("M", args.m_range[0], args.m_range[1], step=args.m_range[2])
    N = trial.suggest_int("N", args.n_range[0], args.n_range[1], step=args.n_range[2])
    K = trial.suggest_int("K", args.k_range[0], args.k_range[1], step=args.k_range[2])

    tflops = benchmark_mm(
        M, N, K, dtype, device, args.num_iterations, args.num_warmup_iterations
    )
    return tflops

def print_progress(study, trial):
    """Print progress during optimization"""
    print(
        f"Trial {trial.number:>6} | {trial.value:6.1f} TFLOPS @ {trial.params['M']}x{trial.params['N']}x{trial.params['K']:<20} | best: {study.best_value:6.1f} TFLOPS",
        end="\r",
    )

def finish(study, start_time):
    """Complete the benchmark and display results"""
    time_delta = time.time() - start_time
    time_str = str(datetime.timedelta(seconds=time_delta)).split(".")[0]
    print("", end="\033[K")
    best_trial = study.best_trial
    best_tflops = best_trial.value
    best_config = f"{best_trial.params['M']}x{best_trial.params['N']}x{best_trial.params['K']} (MxNxK)"
    print(
        f"The best outcome was {best_tflops:.1f}TFLOPS @ {best_config} (tried {len(study.trials)} shapes)"
    )
    print(f"Elapsed time: {time_str}")
    
    # Format the data for database and summary
    perf_data = {
        "tflops": best_tflops,
        "total_time_seconds": time_delta,
    }
    
    config = {
        "best_shape": best_config,
        "m": best_trial.params['M'],
        "n": best_trial.params['N'],
        "k": best_trial.params['K'],
        "trials_count": len(study.trials)
    }
    
    return perf_data, config

def run_benchmark(args):
    """Main entry point for running the MAMF benchmark"""
    # Set up device and dtype
    dtype = torch.bfloat16
    device = arch.device()
    
    # Use the db_file from args - it now has the full path from main.py
    db_path = args.db_file
    
    # Create a SQLite storage with the benchmark-specific database
    storage = RDBStorage(
        url=f"sqlite:///{db_path}", 
        engine_kwargs={"connect_args": {"timeout": 30}}
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
    )
    study.set_metric_names(["TFLOPS"])

    start_time = time.time()

    # Add known best shapes to the study
    known_best_shapes = [
        (6912, 16384, 2048),  # NVIDIA A100 SXM
        (2304, 5120, 1536),   # NVIDIA A100 PCIe
        (6144, 17920, 2816),  # NVIDIA H100 SXM
        (14336, 4096, 4096),  # NVIDIA RTX 4090
        (4352, 13568, 3840),  # AMD MI300X
    ]

    for m, n, k in known_best_shapes:
        study.enqueue_trial({"M": m, "N": n, "K": k})

    # Instead of running multiple iterations with optimize inside,
    # we'll run one trial at a time and check for termination after each
    n_trials_completed = 0
    max_trials = args.num_iterations * args.n_trials
    
    while n_trials_completed < max_trials:
        # Check if termination was requested before starting a new trial
        if should_terminate():
            print("\033[1;33mBenchmark stopped gracefully as requested\033[0m")
            # Save any partial results if needed
            perf_data, config = finish(study, start_time)
            
            # Save results to database if requested
            if hasattr(args, 'db_file') and args.db_file:
                try:
                    benchmark_id = save_benchmark_result(
                        args.db_file, 
                        'mamf', 
                        perf_data, 
                        config
                    )
                    print(f"Benchmark results saved to database (ID: {benchmark_id})")
                except Exception as e:
                    print(f"Error saving to database: {e}")
            
            # Update benchmark summary file if available
            if hasattr(args, 'summary_file') and args.summary_file:
                device_info = str(arch.device_info())
                if update_benchmark_summary(args.summary_file, 'mamf', perf_data, config, device_info):
                    print(f"Benchmark result added to summary: {args.summary_file}")
                update_benchmark_summary('mamf', perf_data, args.summary_file)
                args.summary_updated = True
                    
            return perf_data, config
            
        # Run just one trial
        study.optimize(
            lambda trial: objective(trial, args, dtype, device),
            n_trials=1,
            callbacks=[print_progress]
        )
        
        n_trials_completed += 1

        # Check if termination was requested after each trial
        if TERMINATE_REQUESTED:
            print("\033[1;33mStopping benchmark as requested\033[0m")
            # Save any partial results
            perf_data, config = finish(study, start_time)
            
            # Save results to database if requested
            if hasattr(args, 'db_file') and args.db_file:
                try:
                    benchmark_id = save_benchmark_result(
                        args.db_file, 
                        'mamf', 
                        perf_data, 
                        config
                    )
                    print(f"Benchmark results saved to database (ID: {benchmark_id})")
                except Exception as e:
                    print(f"Error saving to database: {e}")
            
            # Update benchmark summary file if available
            if hasattr(args, 'summary_file') and args.summary_file:
                device_info = str(arch.device_info())
                if update_benchmark_summary(args.summary_file, 'mamf', perf_data, config, device_info):
                    print(f"Benchmark result added to summary: {args.summary_file}")
                update_benchmark_summary('mamf', perf_data, args.summary_file)
                args.summary_updated = True
                    
            return perf_data, config

    # Finish benchmark and return results
    perf_data, config = finish(study, start_time)
    
    # Save results to database if requested
    if hasattr(args, 'db_file') and args.db_file:
        try:
            benchmark_id = save_benchmark_result(
                args.db_file, 
                'mamf', 
                perf_data, 
                config
            )
            print(f"Benchmark results saved to database (ID: {benchmark_id})")
        except Exception as e:
            print(f"Error saving to database: {e}")
    
    # Update benchmark summary file if available
    if hasattr(args, 'summary_file') and args.summary_file:
        device_info = str(arch.device_info())
        if update_benchmark_summary(args.summary_file, 'mamf', perf_data, config, device_info):
            print(f"Benchmark result added to summary: {args.summary_file}")
        update_benchmark_summary('mamf', perf_data, args.summary_file)
        args.summary_updated = True
    
    return perf_data, config