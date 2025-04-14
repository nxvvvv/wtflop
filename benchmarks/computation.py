"""
GPU Computational Task Benchmark Module
Measures computational performance using linear network training.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing
import datetime

from utils.shared_state import TERMINATE_REQUESTED
from utils.arch import get_accelerator_arch
from utils.db_utils import save_benchmark_result, update_benchmark_summary

# Near the top of the file, after imports
if __name__ == "__main__" or multiprocessing.current_process().name == 'MainProcess':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Method might already be set
        pass

def add_benchmark_args(parser):
    """Add benchmark-specific arguments to the parser"""
    group = parser.add_argument_group('Computation Benchmark')
    
    group.add_argument(
        "--epochs", 
        type=int, 
        default=200,
        help="Number of training epochs (default: 200)"
    )
    group.add_argument(
        "--batch-size", 
        type=int, 
        default=2048,
        help="Batch size for training (default: 2048)"
    )
    group.add_argument(
        "--input-size", 
        type=int, 
        default=4096,
        help="Input size for linear model (default: 4096)"
    )
    group.add_argument(
        "--hidden-size", 
        type=int, 
        default=4096,
        help="Hidden layer size for linear model (default: 4096)"
    )
    group.add_argument(
        "--output-size", 
        type=int, 
        default=2000,
        help="Output size for linear model (default: 2000)"
    )
    group.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "float64", "bfloat16"],
        help="Data type for benchmark (default: float16)"
    )
    group.add_argument(
        "--db-file",
        type=str,
        default=None,
        help="Path to database file for saving benchmark results (default: None)"
    )
    group.add_argument(
        "--summary-file",
        type=str,
        default=None,
        help="Path to summary file for updating benchmark results (default: None)"
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

def run_benchmark(args):
    """Run the computational benchmark"""
    print("Running GPU Computational Task Benchmark...")
    
    # Get device information
    arch = get_accelerator_arch()
    if arch.name() != "cuda":
        print("This benchmark requires CUDA-capable GPUs")
        return
    
    # Get benchmark parameters
    epochs = args.epochs
    batch_size = args.batch_size
    input_size = args.input_size
    hidden_size = args.hidden_size
    output_size = args.output_size
    dtype = get_dtype(args.dtype)
    
    # Get device
    device = torch.device('cuda:0')
    
    # Define a simple 2-layer model
    class SimpleModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
            
        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out
    
    # Create model and move to device
    model = SimpleModel(input_size, hidden_size, output_size).to(device=device, dtype=dtype)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Generate random input data and targets
    inputs = torch.randn(batch_size, input_size, device=device, dtype=dtype)
    targets = torch.randn(batch_size, output_size, device=device, dtype=dtype)
    
    # Warm-up
    print("Warming up...")
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    
    # Run training loop
    print(f"Running {epochs} training epochs...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    for epoch in range(epochs):
        if TERMINATE_REQUESTED:
            print(f"Termination requested after {epoch} epochs")
            break
            
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Print progress every 10% of epochs
        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch + 1}/{epochs} completed")
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    epochs_completed = epochs if not TERMINATE_REQUESTED else epoch + 1
    
    # Calculate FLOPS
    # Each forward/backward pass through a linear layer: 2 * in_features * out_features operations
    forward_flops = 2 * (input_size * hidden_size + hidden_size * output_size)
    backward_flops = 2 * forward_flops  # Approximation for backward pass
    total_flops = (forward_flops + backward_flops) * batch_size * epochs_completed
    gflops = total_flops / total_time / 1e9
    
    # Print results
    print(f"\n--- GPU Computational Results ---")
    print(f"Model: 2-layer MLP ({input_size} -> {hidden_size} -> {output_size})")
    print(f"Batch Size: {batch_size}")
    print(f"Data Type: {dtype}")
    print(f"Epochs Completed: {epochs_completed}")
    print(f"Total Time: {total_time:.4f} seconds")
    print(f"Performance: {gflops:.2f} GFLOPS")
    
    # Return performance data and config for logging
    perf_data = {
        "gflops": gflops,
        "total_time_seconds": total_time,
        "epochs_completed": epochs_completed,
    }
    
    config = {
        "batch_size": batch_size,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "dtype": str(dtype).split(".")[-1],
    }
    
    # Clean up
    del model
    del inputs
    del targets
    torch.cuda.empty_cache()
    
    # Save results to database if requested
    if hasattr(args, 'db_file') and args.db_file:
        try:
            benchmark_id = save_benchmark_result(
                args.db_file, 
                'compute', 
                perf_data, 
                config
            )
            print(f"Benchmark results saved to database (ID: {benchmark_id})")
        except Exception as e:
            print(f"Error saving to database: {e}")
    
    # Update benchmark summary file if available
    if hasattr(args, 'summary_file') and args.summary_file:
        device_info = str(get_accelerator_arch().device_info())
        result_data = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "performance": perf_data.get('gflops', 0),
            "config": f"{config['hidden_size']} layers, batch={config['batch_size']}",
            "device": str(device_info)
        }
        update_benchmark_summary('compute', result_data, args.summary_file)
        args.summary_updated = True
    
    return perf_data, config