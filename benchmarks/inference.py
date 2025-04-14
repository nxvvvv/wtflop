"""
GPU Inference Benchmark Module
Measures inference performance with various models.
"""

import time
import torch
import torch.nn as nn
import multiprocessing as mp
import datetime

from utils.shared_state import TERMINATE_REQUESTED
from utils.arch import get_accelerator_arch
from utils.db_utils import save_benchmark_result, update_benchmark_summary

def add_benchmark_args(parser):
    """Add benchmark-specific arguments to the parser"""
    group = parser.add_argument_group('Inference Benchmark')
    group.add_argument(
        "--model-type",
        type=str,
        default="custom",
        choices=["custom", "resnet", "mobilenet", "efficientnet"],
        help="Model type to use for inference benchmark (default: custom)"
    )
    group.add_argument(
        "--model-size", 
        type=int, 
        default=5,
        help="Depth of the custom model (default: 5)"
    )
    group.add_argument(
        "--batch-size", 
        type=int, 
        default=256,
        help="Batch size for inference (default: 256)"
    )
    group.add_argument(
        "--input-size", 
        type=int, 
        default=224,
        help="Input size (default: 224)"
    )
    group.add_argument(
        "--output-size", 
        type=int, 
        default=1000,
        help="Output size for inference (default: 1000)"
    )
    group.add_argument(
        "--iterations", 
        type=int, 
        default=100,
        help="Number of inference iterations (default: 100)"
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
        default="inference.db",  # Provide a default filename instead of None
        help="Path to the database file for saving benchmark results (default: inference.db)"
    )
    group.add_argument(
        "--summary-file",
        type=str,
        default="benchmark_summary.json",  # Provide a default filename
        help="Path to the summary file for updating benchmark results (default: benchmark_summary.json)"
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

def run_inference_on_gpu(logical_gpu_id, model_type, model_size, batch_size, input_size, 
                          output_size, iterations, dtype, return_dict):
    """Run inference benchmark on a specific GPU"""
    try:
        device = torch.device(f'cuda:{logical_gpu_id}')
        torch.cuda.set_device(device)
        
        # Create model based on type
        if model_type.lower() == 'custom':
            # Define custom convolutional model
            class ConvNet(nn.Module):
                def __init__(self, input_channels, num_classes, depth):
                    super(ConvNet, self).__init__()
                    layers = []
                    channels = input_channels
                    for _ in range(depth):
                        layers.append(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1))
                        layers.append(nn.ReLU())
                        layers.append(nn.MaxPool2d(2))
                        channels *= 2
                    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
                    self.features = nn.Sequential(*layers)
                    self.classifier = nn.Linear(channels, num_classes)

                def forward(self, x):
                    x = self.features(x)
                    x = x.view(x.size(0), -1)
                    x = self.classifier(x)
                    return x

            input_channels = 3
            model = ConvNet(input_channels, output_size, model_size).to(device, dtype=dtype)
            
            # Generate input data for CNN
            inputs = torch.randn(batch_size, 3, input_size, input_size, device=device, dtype=dtype)
            
        elif model_type.lower() == 'resnet50':
            from torchvision.models import resnet50
            model = resnet50(pretrained=False).to(device, dtype=dtype)
            
            # Generate input data for ResNet
            inputs = torch.randn(batch_size, 3, input_size, input_size, device=device, dtype=dtype)
            
        elif model_type.lower() == 'bert':
            try:
                from transformers import BertModel, BertConfig
                config = BertConfig()
                model = BertModel(config).to(device, dtype=dtype)
                
                # Generate input data for BERT
                seq_length = input_size
                vocab_size = getattr(model.config, 'vocab_size', 30522)
                inputs = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
            except ImportError:
                print("transformers library not found. Please install it using: pip install transformers")
                return_dict[logical_gpu_id] = None
                return
                
        elif model_type.lower() == 'gpt2':
            try:
                from transformers import GPT2Model, GPT2Config
                config = GPT2Config()
                model = GPT2Model(config).to(device, dtype=dtype)
                
                # Generate input data for GPT-2
                seq_length = input_size
                vocab_size = getattr(model.config, 'vocab_size', 50257)
                inputs = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
            except ImportError:
                print("transformers library not found. Please install it using: pip install transformers")
                return_dict[logical_gpu_id] = None
                return
                
        else:
            print(f"Unsupported model type: {model_type}")
            return_dict[logical_gpu_id] = None
            return
        
        # Set model to evaluation mode
        model.eval()
        
        # Warmup
        with torch.no_grad():
            model(inputs)
        torch.cuda.synchronize()
        
        # Run inference benchmark
        times = []
        for i in range(iterations):
            if TERMINATE_REQUESTED:
                print(f"GPU {logical_gpu_id}: Termination requested after {i} iterations")
                break
                
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                outputs = model(inputs)
            torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
        
        # Calculate statistics
        total_time = sum(times)
        avg_time = total_time / len(times)
        throughput = batch_size / avg_time  # Samples per second
        
        # Store results
        return_dict[logical_gpu_id] = {
            'total_time': total_time,
            'avg_time': avg_time,
            'throughput': throughput,
            'iterations_completed': len(times)
        }
        
        # Clean up
        del model
        del inputs
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error on GPU {logical_gpu_id} during inference: {e}")
        return_dict[logical_gpu_id] = None

def run_benchmark(args):
    """Run the inference benchmark"""
    print("Running GPU Inference Benchmark...")
    
    # Get device information
    arch = get_accelerator_arch()
    if arch.name() != "cuda":
        print("This benchmark requires CUDA-capable GPUs")
        return
    
    # Get benchmark parameters
    model_type = args.model_type
    model_size = args.model_size
    batch_size = args.batch_size
    input_size = args.input_size
    output_size = args.output_size
    iterations = args.iterations
    dtype = get_dtype(args.dtype)
    
    # Get logical GPU IDs
    device_count = torch.cuda.device_count()
    if device_count == 0:
        print("No CUDA devices found")
        return
    
    # Adjust batch size per GPU
    batch_size_per_gpu = batch_size // device_count
    if batch_size_per_gpu == 0:
        batch_size_per_gpu = 1
        print(f"Warning: Batch size {batch_size} is too small for {device_count} GPUs.")
        print(f"Setting minimum batch size of 1 per GPU (total {device_count}).")
    
    # Launch parallel inference processes
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    
    for logical_id in range(device_count):
        p = mp.Process(
            target=run_inference_on_gpu, 
            args=(
                logical_id, 
                model_type, 
                model_size, 
                batch_size_per_gpu, 
                input_size, 
                output_size, 
                iterations, 
                dtype, 
                return_dict
            )
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Process results
    valid_results = [v for v in return_dict.values() if v is not None]
    if not valid_results:
        print("Error: No valid inference results collected")
        return
    
    # Calculate aggregated metrics
    total_throughput = sum(r['throughput'] for r in valid_results)
    avg_latency = sum(r['avg_time'] for r in valid_results) / len(valid_results)
    total_time = max(r['total_time'] for r in valid_results)  # Max time among all GPUs
    total_iterations = sum(r['iterations_completed'] for r in valid_results)
    
    # Print results
    print(f"\n--- GPU Inference Results ---")
    print(f"Model: {model_type}")
    print(f"Model Size: {model_size}")
    print(f"Data Type: {dtype}")
    print(f"Batch Size: {batch_size} ({batch_size_per_gpu} per GPU)")
    print(f"Input Size: {input_size}")
    print(f"Number of GPUs: {device_count}")
    
    for i, result in enumerate(valid_results):
        print(f"GPU {i}: {result['throughput']:.2f} samples/sec, {result['avg_time']*1000:.2f} ms/batch")
    
    print(f"Total Throughput: {total_throughput:.2f} samples/sec")
    print(f"Average Latency: {avg_latency*1000:.2f} ms/batch")
    print(f"Total Time: {total_time:.4f} seconds")
    print(f"Total Iterations: {total_iterations}")
    
    # Return performance data and config for logging
    perf_data = {
        "throughput_samples_per_sec": total_throughput,
        "avg_latency_ms": avg_latency * 1000,
        "total_time_seconds": total_time,
    }
    
    config = {
        "model_type": model_type,
        "model_size": model_size,
        "batch_size": batch_size,
        "input_size": input_size,
        "output_size": output_size,
        "dtype": str(dtype).split(".")[-1],
        "num_gpus": device_count,
    }
    
    # Save results to database if requested
    if hasattr(args, 'db_file') and args.db_file:
        try:
            benchmark_id = save_benchmark_result(
                args.db_file, 
                'inference', 
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
            "performance": perf_data.get('throughput_samples_per_sec', 0),
            "config": f"{args.model_type}, batch={args.batch_size}, input={args.input_size}",
            "device": str(device_info)
        }
        update_benchmark_summary('inference', result_data, args.summary_file)
        args.summary_updated = True
    
    return perf_data, config