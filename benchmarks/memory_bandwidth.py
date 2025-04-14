"""
Memory Bandwidth Benchmark Module
Measures both GPU and system memory bandwidth.
"""

import time
import torch
import numpy as np

from utils.shared_state import TERMINATE_REQUESTED
from utils.arch import get_accelerator_arch
from utils.db_utils import save_benchmark_result, update_benchmark_summary
import multiprocessing
# Near the top of the file, after imports
if __name__ == "__main__" or multiprocessing.current_process().name == 'MainProcess':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Method might already be set
        pass
def add_benchmark_args(parser):
    """Add benchmark-specific arguments to the parser"""
    group = parser.add_argument_group('Memory Bandwidth Benchmarks')
    
    group.add_argument(
        "--benchmark-type",
        type=str,
        choices=["gpu", "system", "both"],
        default="both",
        help="Type of memory bandwidth benchmark to run (default: both)"
    )
    group.add_argument(
        "--gpu-mem-size-gb", 
        type=float, 
        default=5.0,
        help="Data size in GB for GPU memory bandwidth benchmark (default: 5.0)"
    )
    group.add_argument(
        "--system-mem-size-mb", 
        type=int, 
        default=1024,
        help="Data size in MB for system memory bandwidth benchmark (default: 1024)"
    )
    group.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "float64", "bfloat16"],
        help="Data type for benchmark (default: float16)"
    )

def get_torch_dtype(dtype_str):
    """Convert string dtype to torch dtype"""
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float16)

def get_numpy_dtype(dtype_str):
    """Convert string dtype to numpy dtype"""
    dtype_map = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
        "bfloat16": np.float16,  # NumPy doesn't have bfloat16
    }
    return dtype_map.get(dtype_str, np.float32)

def benchmark_gpu_memory_bandwidth(args):
    """Measure GPU internal memory bandwidth"""
    print("Running GPU Memory Bandwidth Benchmark...")
    
    # Get device information
    arch = get_accelerator_arch()
    if arch.name() != "cuda":
        print("This benchmark requires CUDA-capable GPUs")
        return None, None
    
    # Get benchmark parameters
    data_size_gb = args.gpu_mem_size_gb
    dtype = get_torch_dtype(args.dtype)
    device = torch.device('cuda:0')
    
    # Convert GB to bytes
    data_size_bytes = data_size_gb * 1024 * 1024 * 1024
    element_size = torch.tensor([], dtype=dtype).element_size()
    num_elements = int(data_size_bytes // element_size)
    
    try:
        # Generate source data on GPU
        print(f"Generating {data_size_gb:.2f} GB of data on GPU...")
        src_tensor = torch.randn((num_elements,), device=device, dtype=dtype)
        
        # Warm-up
        torch.cuda.synchronize()
        _ = src_tensor.clone()
        torch.cuda.synchronize()
        
        # Measure memory copy bandwidth
        print("Measuring memory bandwidth...")
        torch.cuda.synchronize()
        start_time = time.time()
        dest_tensor = src_tensor.clone()
        torch.cuda.synchronize()
        end_time = time.time()
        
        copy_time = end_time - start_time
        bandwidth_gb_per_sec = data_size_gb / copy_time if copy_time > 0 else float('inf')
        
        # Print results
        print(f"\n--- GPU Memory Bandwidth Results ---")
        print(f"Data Size: {data_size_gb:.2f} GB")
        print(f"Data Type: {dtype}")
        print(f"Copy Time: {copy_time:.6f} seconds")
        print(f"Memory Bandwidth: {bandwidth_gb_per_sec:.2f} GB/s")
        
        # Clean up
        del src_tensor
        del dest_tensor
        torch.cuda.empty_cache()
        
        # Return performance data and config
        perf_data = {
            "gpu_bandwidth_GB_per_s": bandwidth_gb_per_sec,
            "gpu_copy_time_seconds": copy_time,
        }
        
        config = {
            "gpu_data_size_gb": data_size_gb,
            "dtype": str(dtype).split(".")[-1],
        }
        
        return perf_data, config
        
    except RuntimeError as e:
        print(f"Error during GPU memory bandwidth benchmark: {e}")
        # Try with a smaller size if OOM occurs
        if "out of memory" in str(e).lower() and data_size_gb > 0.5:
            print(f"Retrying with half the data size ({data_size_gb/2:.2f} GB)...")
            args.gpu_mem_size_gb = data_size_gb / 2
            return benchmark_gpu_memory_bandwidth(args)
        return None, None

def benchmark_system_memory_bandwidth(args):
    """Measure system memory bandwidth"""
    print("Running System Memory Bandwidth Benchmark...")
    
    # Get benchmark parameters
    memory_size_mb = args.system_mem_size_mb
    dtype_str = args.dtype
    np_dtype = get_numpy_dtype(dtype_str)
    
    try:
        # Convert MB to bytes
        data_size = memory_size_mb * 1024 * 1024
        element_size = np.dtype(np_dtype).itemsize
        array_size = data_size // element_size
        
        # Generate source data
        print(f"Generating {memory_size_mb:.2f} MB of data in system memory...")
        src_array = np.random.rand(array_size).astype(np_dtype)
        
        # Warm-up
        _ = np.copy(src_array)
        
        # Measure memory copy bandwidth
        print("Measuring memory bandwidth...")
        start_time = time.time()
        dest_array = np.copy(src_array)
        end_time = time.time()
        
        copy_time = end_time - start_time
        bandwidth_gb_per_sec = (data_size / copy_time) / 1e9 if copy_time > 0 else float('inf')
        
        # Print results
        print(f"\n--- System Memory Bandwidth Results ---")
        print(f"Data Size: {memory_size_mb:.2f} MB")
        print(f"Data Type: {np_dtype}")
        print(f"Copy Time: {copy_time:.6f} seconds")
        print(f"Memory Bandwidth: {bandwidth_gb_per_sec:.2f} GB/s")
        
        # Return performance data and config
        perf_data = {
            "system_bandwidth_GB_per_s": bandwidth_gb_per_sec,
            "system_copy_time_seconds": copy_time,
        }
        
        config = {
            "system_data_size_mb": memory_size_mb,
            "dtype": str(np_dtype),
        }
        
        return perf_data, config
        
    except Exception as e:
        print(f"Error during system memory bandwidth benchmark: {e}")
        return None, None

def run_benchmark(args):
    """Run memory bandwidth benchmarks based on specified type"""
    benchmark_type = args.benchmark_type
    results = {}
    
    if benchmark_type in ["gpu", "both"]:
        gpu_perf, gpu_config = benchmark_gpu_memory_bandwidth(args)
        if gpu_perf:
            results["gpu"] = (gpu_perf, gpu_config)
    
    if benchmark_type in ["system", "both"]:
        system_perf, system_config = benchmark_system_memory_bandwidth(args)
        if system_perf:
            results["system"] = (system_perf, system_config)
    
    # Return combined results
    if not results:
        return None
    
    # Combine all results into a single return value
    combined_perf = {}
    combined_config = {}
    
    for key, (perf, config) in results.items():
        combined_perf.update(perf)
        combined_config.update(config)
    
    # Save results to database if requested
    if hasattr(args, 'db_file') and args.db_file:
        try:
            benchmark_id = save_benchmark_result(
                args.db_file, 
                'membw', 
                combined_perf, 
                combined_config
            )
            print(f"Benchmark results saved to database (ID: {benchmark_id})")
        except Exception as e:
            print(f"Error saving to database: {e}")
    
    # Update benchmark summary file if available
    if hasattr(args, 'summary_file') and args.summary_file:
        device_info = str(get_accelerator_arch().device_info())
        if update_benchmark_summary(args.summary_file, 'membw', combined_perf, combined_config, device_info):
            print(f"Benchmark result added to summary: {args.summary_file}")
        update_benchmark_summary('membw', combined_perf, args.summary_file)
        args.summary_updated = True
    
    return combined_perf, combined_config