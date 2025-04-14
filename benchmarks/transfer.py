"""
GPU Transfer Benchmark Module
Measures GPU-to-CPU and GPU-to-GPU transfer speeds.
"""

import time
import torch
import multiprocessing as mp
import multiprocessing
import platform

# Near the top of the file, after imports
if __name__ == "__main__" or multiprocessing.current_process().name == 'MainProcess':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Method might already be set
        pass

from utils.shared_state import TERMINATE_REQUESTED
from utils.arch import get_accelerator_arch
from utils.db_utils import save_benchmark_result, update_benchmark_summary

def add_benchmark_args(parser):
    """Add benchmark-specific arguments to the parser"""
    group = parser.add_argument_group('Transfer Benchmark')
    group.add_argument(
        "--benchmark-type",
        type=str,
        default="both",
        choices=["gpu-to-cpu", "gpu-to-gpu", "both"],
        help="Type of transfer benchmark to run (default: both)"
    )
    group.add_argument(
        "--data-size-gb", 
        type=float, 
        default=None,  # Change to None so we can auto-calculate
        help="Data size in GB for transfer benchmarks (default: auto-calculated based on available memory)"
    )
    group.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "float64", "bfloat16"],
        help="Data type for benchmark (default: float16)"
    )
    group.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations for GPU-to-GPU transfer (default: 10)"
    )

def get_dtype(dtype_str):
    """Convert string dtype to torch dtype"""
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float16)

def run_gpu_to_cpu_transfer_on_gpu(logical_gpu_id, num_elements, dtype, return_dict):
    """Run GPU-to-CPU transfer benchmark on a specific GPU"""
    try:
        device = torch.device(f'cuda:{logical_gpu_id}')
        torch.cuda.set_device(device)
        
        # Generate test data on GPU
        data = torch.randn(num_elements, device=device, dtype=dtype)
        torch.cuda.synchronize()
        
        # Measure transfer time
        start_time = time.time()
        cpu_data = data.cpu()
        torch.cuda.synchronize()
        end_time = time.time()
        
        transfer_time = end_time - start_time
        return_dict[logical_gpu_id] = transfer_time
        
        # Clean up
        del data
        del cpu_data
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error on logical GPU {logical_gpu_id} during GPU to CPU transfer: {e}")
        return_dict[logical_gpu_id] = None

def run_gpu_to_gpu_transfer(logical_gpu0_id, logical_gpu1_id, num_elements, iterations, dtype, return_dict):
    """Run GPU-to-GPU transfer benchmark between two GPUs"""
    try:
        device0 = torch.device(f'cuda:{logical_gpu0_id}')
        device1 = torch.device(f'cuda:{logical_gpu1_id}')
        
        # Set current device 
        torch.cuda.set_device(device0)
        
        # Create source and destination tensors
        src_tensor = torch.randn(num_elements, device=device0, dtype=dtype)
        dest_tensor = torch.empty(num_elements, device=device1, dtype=dtype)

        # Warm-up
        torch.cuda.synchronize()
        dest_tensor.copy_(src_tensor)
        torch.cuda.synchronize()

        # Measure transfer time over multiple iterations
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(iterations):
            dest_tensor.copy_(src_tensor)
        torch.cuda.synchronize()
        end_time = time.time()

        total_time = end_time - start_time
        average_time = total_time / iterations
        return_dict['copy_time'] = average_time
        
        # Clean up
        del src_tensor
        del dest_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error during GPU to GPU transfer: {e}")
        return_dict['copy_time'] = None

def benchmark_gpu_to_cpu(args):
    """Run the GPU-to-CPU transfer benchmark"""
    print("Running GPU-to-CPU Transfer Benchmark...")
    
    # Get device information
    arch = get_accelerator_arch()
    if arch.name() != "cuda":
        print("This benchmark requires CUDA-capable GPUs")
        return None, None
    
    # Get benchmark parameters
    data_size_gb = args.data_size_gb
    dtype = get_dtype(args.dtype)
    
    # Calculate elements based on data type size
    element_size = torch.tensor([], dtype=dtype).element_size()
    
    # Get logical GPU IDs
    device_count = torch.cuda.device_count()
    if device_count == 0:
        print("No CUDA devices found")
        return None, None
    
    # Calculate elements per GPU
    num_elements = int((data_size_gb * 1e9) / element_size)
    num_elements_per_gpu = num_elements // device_count
    
    # Launch parallel transfer processes
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    
    for logical_id in range(device_count):
        p = mp.Process(
            target=run_gpu_to_cpu_transfer_on_gpu, 
            args=(logical_id, num_elements_per_gpu, dtype, return_dict)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Process results
    transfer_times = [t for t in return_dict.values() if t is not None]
    if not transfer_times:
        print("Error: No valid transfer times collected")
        return None, None
    
    # Calculate metrics
    max_time = max(transfer_times)
    data_size_bytes = num_elements_per_gpu * element_size * len(transfer_times)
    data_size_gb_actual = data_size_bytes / 1e9
    transfer_bandwidth = data_size_gb_actual / max_time
    
    # Print results
    print(f"\n--- GPU-to-CPU Transfer Results ---")
    print(f"Data Size: {data_size_gb_actual:.2f} GB total")
    print(f"Data Type: {dtype}")
    print(f"Number of GPUs: {len(transfer_times)}")
    print(f"Transfer Bandwidth: {transfer_bandwidth:.2f} GB/s")
    print(f"Total Time: {max_time:.4f} seconds")
    
    # Return performance data and config
    perf_data = {
        "bandwidth_GB_per_s": transfer_bandwidth,
        "time_seconds": max_time,
    }
    
    config = {
        "data_size_gb": data_size_gb_actual,
        "dtype": str(dtype).split(".")[-1],
        "num_gpus": len(transfer_times),
    }
    
    return perf_data, config

def benchmark_gpu_to_gpu(args):
    """Run the GPU-to-GPU transfer benchmark"""
    print("Running GPU-to-GPU Transfer Benchmark...")
    
    # Get device information
    arch = get_accelerator_arch()
    if arch.name() != "cuda":
        print("This benchmark requires CUDA-capable GPUs")
        return None, None
    
    # Get logical GPU IDs
    device_count = torch.cuda.device_count()
    if device_count < 2:
        print("This benchmark requires at least 2 CUDA-capable GPUs")
        return None, None
    
    # Get benchmark parameters
    data_size_gb = args.data_size_gb
    dtype = get_dtype(args.dtype)
    iterations = args.iterations
    
    # Calculate elements based on data type size
    element_size = torch.tensor([], dtype=dtype).element_size()
    num_elements = int((data_size_gb * 1e9) / element_size)
    
    # Launch the GPU-to-GPU transfer process
    manager = mp.Manager()
    return_dict = manager.dict()
    
    # Test transfer between first two GPUs (0->1)
    p = mp.Process(
        target=run_gpu_to_gpu_transfer, 
        args=(0, 1, num_elements, iterations, dtype, return_dict)
    )
    p.start()
    p.join()
    
    # Process results
    copy_time = return_dict.get('copy_time')
    if copy_time is None:
        print("Error: Failed to collect copy time")
        return None, None
    
    # Calculate bandwidth
    data_size_bytes = num_elements * element_size
    data_size_gb_actual = data_size_bytes / 1e9
    bandwidth = data_size_gb_actual / copy_time
    
    # Print results
    print(f"\n--- GPU-to-GPU Transfer Results ---")
    print(f"Data Size: {data_size_gb_actual:.2f} GB")
    print(f"Data Type: {dtype}")
    print(f"Iterations: {iterations}")
    print(f"Average Copy Time: {copy_time:.6f} seconds per iteration")
    print(f"Transfer Bandwidth: {bandwidth:.2f} GB/s")
    
    # Return performance data and config
    perf_data = {
        "bandwidth_GB_per_s": bandwidth,
        "time_per_iteration_seconds": copy_time,
        "total_time_seconds": copy_time * iterations,
    }
    
    config = {
        "data_size_gb": data_size_gb_actual,
        "dtype": str(dtype).split(".")[-1],
        "iterations": iterations,
    }
    
    return perf_data, config

def run_benchmark(args):
    """Run the transfer benchmark"""
    # Auto-calculate the data size if not specified
    if args.data_size_gb is None:
        # Get available GPU memory (in GB)
        free_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) * 0.8  # 80% of total GPU memory
        
        # Get available system memory (in GB)
        if platform.system() == "Windows":
            import psutil
            free_system_memory = psutil.virtual_memory().available / (1024**3)
        else:
            # For Linux/Unix systems
            import os
            mem_info = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            free_system_memory = mem_info / (1024**3) * 0.8  # 80% of total system memory
        
        # Use the smaller of the two memory values (leave 20% buffer)
        args.data_size_gb = min(free_gpu_memory, free_system_memory)
        print(f"Auto-calculated data size: {args.data_size_gb:.2f} GB based on available memory")
    else:
        # If manually specified, verify it's not too large
        free_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) * 0.8
        if args.data_size_gb > free_gpu_memory:
            print(f"Warning: Requested {args.data_size_gb:.2f} GB exceeds available GPU memory. Limiting to {free_gpu_memory:.2f} GB")
            args.data_size_gb = free_gpu_memory
    
    benchmark_type = args.benchmark_type
    results = {}
    
    if benchmark_type in ["gpu-to-cpu", "both"]:
        gpu_to_cpu_perf, gpu_to_cpu_config = benchmark_gpu_to_cpu(args)
        if gpu_to_cpu_perf:
            results["gpu_to_cpu"] = (gpu_to_cpu_perf, gpu_to_cpu_config)
    
    if benchmark_type in ["gpu-to-gpu", "both"]:
        # Check if we have at least 2 GPUs
        if torch.cuda.device_count() >= 2:
            gpu_to_gpu_perf, gpu_to_gpu_config = benchmark_gpu_to_gpu(args)
            if gpu_to_gpu_perf:
                results["gpu_to_gpu"] = (gpu_to_gpu_perf, gpu_to_gpu_config)
        else:
            print("GPU-to-GPU transfer requires at least 2 GPUs - skipping this benchmark")
    
    # Return combined results
    if not results:
        return None
    
    # Combine all results into a single return value
    combined_perf = {}
    combined_config = {}
    
    for key, (perf, config) in results.items():
        # Add benchmark type prefix to keys
        for perf_key, perf_value in perf.items():
            combined_perf[f"{key}_{perf_key}"] = perf_value
        
        for config_key, config_value in config.items():
            combined_config[f"{key}_{config_key}"] = config_value
    
    # Save results to database if requested
    if hasattr(args, 'db_file') and args.db_file:
        try:
            benchmark_id = save_benchmark_result(
                args.db_file, 
                'transfer', 
                combined_perf, 
                combined_config
            )
            print(f"Benchmark results saved to database (ID: {benchmark_id})")
        except Exception as e:
            print(f"Error saving to database: {e}")
    
    # Update benchmark summary file if available
    if hasattr(args, 'summary_file') and args.summary_file:
        device_info = str(get_accelerator_arch().device_info())
        if update_benchmark_summary(args.summary_file, 'transfer', combined_perf, combined_config, device_info):
            print(f"Benchmark result added to summary: {args.summary_file}")
        update_benchmark_summary('transfer', combined_perf, args.summary_file)
        args.summary_updated = True
    
    return combined_perf, combined_config