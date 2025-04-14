"""
GPU Data Generation Benchmark Module
Measures how quickly a GPU can generate random data.
"""

import time
import torch
import multiprocessing as mp
import multiprocessing

# Set multiprocessing start method
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Method might already be set in parent process
    pass

from utils.shared_state import TERMINATE_REQUESTED
from utils.arch import get_accelerator_arch
from utils.db_utils import save_benchmark_result, update_benchmark_summary

def add_benchmark_args(parser):
    """Add benchmark-specific arguments to the parser"""
    group = parser.add_argument_group('Data Generation Benchmark')
    group.add_argument(
        "--data-size-gb", 
        type=float, 
        default=5.0,
        help="Data size in GB for GPU data generation benchmark (default: 5.0)"
    )
    group.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "float64", "bfloat16"],
        help="Data type for benchmark (default: float16)"
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

def run_data_generation_on_gpu(logical_gpu_id, num_elements, dtype, return_dict):
    """Run the data generation benchmark on a specific GPU"""
    try:
        device = torch.device(f'cuda:{logical_gpu_id}')
        torch.cuda.set_device(device)
        start_time = time.time()
        tensor = torch.randn(num_elements, device=device, dtype=dtype)
        torch.cuda.synchronize()
        end_time = time.time()
        gen_time = end_time - start_time
        return_dict[logical_gpu_id] = gen_time
        
        # Clean up
        del tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error on logical GPU {logical_gpu_id} during data generation: {e}")
        return_dict[logical_gpu_id] = None

def run_benchmark(args):
    """Run the GPU data generation benchmark"""
    # Set the multiprocessing start method to 'spawn'
    if __name__ == '__main__' or multiprocessing.get_start_method() != 'spawn':
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            # Method might already be set
            pass

    print("Running GPU Data Generation Benchmark...")
    
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
    num_elements = int((data_size_gb * 1e9) / element_size)
    
    # Get logical GPU IDs
    device_count = torch.cuda.device_count()
    if device_count == 0:
        print("No CUDA devices found")
        return None, None
    
    num_elements_per_gpu = num_elements // device_count
    
    # Launch parallel data generation processes
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    
    for logical_id in range(device_count):
        p = mp.Process(
            target=run_data_generation_on_gpu, 
            args=(logical_id, num_elements_per_gpu, dtype, return_dict)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Process results
    gen_times = [t for t in return_dict.values() if t is not None]
    if not gen_times:
        print("Error: No valid generation times collected")
        return None, None
    
    # Calculate metrics
    max_time = max(gen_times)
    data_size_bytes_per_gpu = num_elements_per_gpu * element_size
    data_size_gb_per_gpu = data_size_bytes_per_gpu / 1e9
    total_data_size_gb = data_size_gb_per_gpu * device_count
    
    # Calculate bandwidth for each GPU and the total
    bandwidths = []
    for gen_time in gen_times:
        bandwidth = data_size_gb_per_gpu / gen_time if gen_time > 0 else 0
        bandwidths.append(bandwidth)
    
    total_bandwidth = sum(bandwidths)
    
    # Print results
    print(f"\n--- GPU Data Generation Results ---")
    print(f"Data Size: {total_data_size_gb:.2f} GB ({data_size_gb_per_gpu:.2f} GB per GPU)")
    print(f"Data Type: {dtype}")
    print(f"Number of GPUs: {device_count}")
    
    for i, (gen_time, bandwidth) in enumerate(zip(gen_times, bandwidths)):
        print(f"GPU {i}: {bandwidth:.2f} GB/s ({gen_time:.4f} seconds)")
    
    print(f"Total Bandwidth: {total_bandwidth:.2f} GB/s")
    print(f"Total Time: {max_time:.4f} seconds")
    
    # Return performance data and config for logging
    perf_data = {
        "total_bandwidth_GB_per_s": total_bandwidth,
        "time_seconds": max_time,
        "per_gpu_bandwidths_GB_per_s": bandwidths,
    }
    
    config = {
        "data_size_gb": total_data_size_gb,
        "dtype": str(dtype).split(".")[-1],
        "num_gpus": device_count,
    }
    
    # Save results to database if requested
    if hasattr(args, 'db_file') and args.db_file:
        try:
            benchmark_id = save_benchmark_result(
                args.db_file, 
                'datagen', 
                perf_data, 
                config
            )
            print(f"Benchmark results saved to database (ID: {benchmark_id})")
        except Exception as e:
            print(f"Error saving to database: {e}")
    
    # Update benchmark summary file if available
    if hasattr(args, 'summary_file') and args.summary_file:
        device_info = str(arch.device_info())
        if update_benchmark_summary(args.summary_file, 'datagen', perf_data, config, device_info):
            print(f"Benchmark result added to summary: {args.summary_file}")
        update_benchmark_summary('datagen', perf_data, args.summary_file)
        args.summary_updated = True
    
    return perf_data, config