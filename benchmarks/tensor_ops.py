import time
import datetime
import numpy as np
import os
import sys
import torch
import optuna
from optuna.storages import RDBStorage
import warnings
from pathlib import Path
import pandas as pd
import gc
from utils.shared_state import TERMINATE_REQUESTED, should_terminate, set_trial_status
from utils.arch import get_accelerator_arch
from utils.db_utils import save_benchmark_result, update_benchmark_summary

warnings.filterwarnings("ignore", category=UserWarning)

# Get architecture singleton
arch = get_accelerator_arch()

def add_benchmark_args(parser):
    """Add Tensor-specific arguments to the ArgumentParser"""
    parser.add_argument(
        "--batch_sizes",
        nargs="+",
        type=int,
        default=[1, 8, 16, 32, 64, 128],
        help="Batch sizes to test"
    )
    parser.add_argument(
        "--tensor_sizes",
        nargs="+",
        type=int,
        default=[128, 256, 512, 1024, 2048],
        help="Size of tensor dimensions (N) to test for NxN operations"
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default=["float16", "bfloat16"],
        help="Data types to test"
    )
    parser.add_argument(
        "--operations",
        nargs="+",
        type=str,
        choices=["matmul", "conv2d", "attention", "layernorm", "softmax", "all"],
        default=["all"],
        help="Tensor operations to benchmark"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of iterations for timing each operation"
    )
    parser.add_argument(
        "--num_warmup_iterations",
        type=int,
        default=50,
        help="Number of warmup iterations before timing"
    )
    parser.add_argument(
        "--optimize_config",
        action="store_true",
        help="Use Optuna to find optimal tensor configurations"
    )
    parser.add_argument(
        "--n_trials", 
        type=int, 
        default=100, 
        help="Number of trials for Optuna optimization"
    )
    parser.add_argument(
        "--study_name", 
        type=str, 
        default="tensor_ops_study", 
        help="Name of the Optuna study"
    )
    parser.add_argument(
        "--db_file",
        type=str,
        default="tensor_ops.db",
        help="Database file to store Optuna results"
    )
    parser.add_argument(
        "--memory_safety",
        action="store_true",
        default=True,
        help="Enable memory safety features to prevent OOM errors"
    )
    parser.add_argument(
        "--memory_buffer",
        type=float,
        default=0.3,
        help="Memory buffer to reserve (fraction of total GPU memory)"
    )
    # Add new database-related arguments
    parser.add_argument(
        "--analyze_db",
        action="store_true",
        help="Analyze existing benchmark results in database instead of running new benchmarks"
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        help="Directory to save analysis report (default: results/reports/tensor_analysis_DATE)"
    )
    parser.add_argument(
        "--filter_operation",
        type=str,
        choices=["matmul", "conv2d", "attention", "layernorm", "softmax", "all"],
        help="Filter database analysis by operation"
    )
    parser.add_argument(
        "--filter_dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Filter database analysis by data type"
    )
    parser.add_argument(
        "--min_tflops",
        type=float,
        help="Filter database analysis by minimum TFLOPS value"
    )
    # Add database-specific arguments if not already present
    database_group = parser.add_argument_group('Database Options')
    database_group.add_argument(
        "--db-file",
        type=str,
        default=None,
        help="Path to database file for saving benchmark results (default: None)"
    )
    database_group.add_argument(
        "--summary-file",
        type=str,
        default=None,
        help="Path to summary file for updating benchmark results (default: None)"
    )

def get_dtype(dtype_str):
    """Convert string dtype to torch dtype"""
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

def estimate_memory_usage(B, N, op_type, dtype_str):
    """
    Estimate memory usage for a tensor operation
    Returns memory estimate in bytes
    """
    # Get bytes per element based on dtype
    if dtype_str == "float32":
        bytes_per_element = 4
    else:  # float16 or bfloat16
        bytes_per_element = 2
        
    # Estimate memory usage based on operation type
    if op_type == "matmul":
        # A: B×N×N, B: B×N×N, output: B×N×N, + workspace
        return 3 * B * N * N * bytes_per_element * 1.5  # Adding 50% for workspace
    
    elif op_type == "conv2d":
        in_channels = 3
        out_channels = min(N, 64)
        kernel_size = min(7, N//8) if N > 32 else 3
        # Input: B×C×N×N, weights: out_c×in_c×k×k, output: B×out_c×N×N, + workspace
        return (B * in_channels * N * N + out_channels * in_channels * kernel_size**2 + 
                B * out_channels * N * N) * bytes_per_element * 2.0  # Double for workspace
    
    elif op_type == "attention":
        seq_len = N
        hidden_dim = min(N, 512)
        # Q, K, V: 3×B×seq_len×hidden_dim, + attention: B×seq_len×seq_len, + output
        return (3 * B * seq_len * hidden_dim + B * seq_len * seq_len + 
                B * seq_len * hidden_dim) * bytes_per_element * 2.0
    
    elif op_type == "layernorm" or op_type == "softmax":
        # Input and output tensors + small overhead
        return 2 * B * N * N * bytes_per_element * 1.2
    
    # Default case
    return B * N * N * bytes_per_element * 4  # Conservative estimate

def check_memory_feasibility(B, N, op_type, dtype_str, device):
    """
    Check if an operation is feasible with current memory constraints
    Returns: (is_feasible, estimated_memory_bytes)
    """
    # Get available memory
    with torch.cuda.device(device):
        total_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = torch.cuda.memory_allocated(device)
        available_memory = total_memory - reserved_memory
        
        # Estimate memory needed for operation
        estimated_memory = estimate_memory_usage(B, N, op_type, dtype_str)
        
        # Check if feasible (leave 20% buffer)
        is_feasible = (estimated_memory <= available_memory * 0.8)
        
        return is_feasible, estimated_memory

def clean_up_memory():
    """Force garbage collection and clear CUDA cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def benchmark_matmul(B, N, dtype, device, num_iterations, num_warmup_iterations):
    """Benchmark matrix multiplication with attention-like shapes"""
    # Create tensors similar to attention computation
    start = arch.event(enable_timing=True)
    end = arch.event(enable_timing=True)
    
    try:
        # Create query and key matrices (B, N, N)
        A = torch.randn(B, N, N, dtype=dtype, device=device)
        B_tensor = torch.randn(B, N, N, dtype=dtype, device=device)
        
        # The output should be (B, N, N)
        times = np.zeros(num_iterations + num_warmup_iterations)
        
        for i in range(num_warmup_iterations + num_iterations):
            arch.clear_cache()
            with torch.no_grad():
                start.record()
                torch.bmm(A, B_tensor)  # Batch matrix multiplication
                end.record()
            arch.synchronize()
            times[i] = start.elapsed_time(end)
        
        times = times[num_warmup_iterations:]
        elapsed_time = np.median(times) / 1000  # median time in seconds
        
        # Calculate throughput: (2*B*N^3) FLOPs for BMM
        flops = 2 * B * N**3
        tflops = flops / (elapsed_time * 10**12)
        
        return tflops
    
    finally:
        # Clean up tensors to free memory
        del A, B_tensor
        clean_up_memory()

def benchmark_conv2d(B, N, dtype, device, num_iterations, num_warmup_iterations):
    """Benchmark 2D convolution operation"""
    start = arch.event(enable_timing=True)
    end = arch.event(enable_timing=True)
    
    try:
        # Adjust N to make sure it's appropriate for convolution
        H, W = N, N
        in_channels = min(3, max(1, N // 256))  # Scale input channels down for large N
        out_channels = min(N // 16, 64)  # Scale output channels based on N
        kernel_size = min(7, N//32) if N > 32 else 3  # Smaller kernel for large inputs
        
        # Create input tensor (B, C, H, W)
        x = torch.randn(B, in_channels, H, W, dtype=dtype, device=device)
        
        # Create convolution layer
        conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            padding=kernel_size//2, device=device
        ).to(dtype=dtype)
        
        times = np.zeros(num_iterations + num_warmup_iterations)
        
        for i in range(num_warmup_iterations + num_iterations):
            arch.clear_cache()
            with torch.no_grad():
                start.record()
                conv(x)
                end.record()
            arch.synchronize()
            times[i] = start.elapsed_time(end)
        
        times = times[num_warmup_iterations:]
        elapsed_time = np.median(times) / 1000  # seconds
        
        # Approximate FLOPs for convolution
        # FLOPs = 2 * B * out_channels * in_channels * kernel_size^2 * H * W
        flops = 2 * B * out_channels * in_channels * kernel_size**2 * H * W
        tflops = flops / (elapsed_time * 10**12)
        
        return tflops
    
    finally:
        # Clean up tensors to free memory
        del x, conv
        clean_up_memory()

def benchmark_attention(B, N, dtype, device, num_iterations, num_warmup_iterations):
    """Benchmark self-attention operation (Q*K^T, then softmax, then V)"""
    start = arch.event(enable_timing=True)
    end = arch.event(enable_timing=True)
    
    try:
        # Adjust dimensions to be appropriate for attention
        seq_len = min(N, 2048)  # Cap sequence length to avoid OOM
        hidden_dim = min(N, 512)  # Scale hidden dimension based on N
        num_heads = min(8, max(1, hidden_dim // 64))  # Scale number of heads
        
        # Create Q, K, V tensors
        Q = torch.randn(B, seq_len, hidden_dim, dtype=dtype, device=device)
        K = torch.randn(B, seq_len, hidden_dim, dtype=dtype, device=device)
        V = torch.randn(B, seq_len, hidden_dim, dtype=dtype, device=device)
        
        # Create a multi-head attention module
        attn = torch.nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, device=device
        ).to(dtype=dtype)
        
        times = np.zeros(num_iterations + num_warmup_iterations)
        
        for i in range(num_warmup_iterations + num_iterations):
            arch.clear_cache()
            with torch.no_grad():
                start.record()
                attn(Q, K, V)[0]  # Get output tensor only
                end.record()
            arch.synchronize()
            times[i] = start.elapsed_time(end)
        
        times = times[num_warmup_iterations:]
        elapsed_time = np.median(times) / 1000  # seconds
        
        # Approximate FLOPs for attention
        # QK^T = 2*B*seq_len^2*hidden_dim
        # Softmax ~ seq_len^2
        # Attention*V = 2*B*seq_len^2*hidden_dim
        flops = 4 * B * seq_len**2 * hidden_dim + B * seq_len**2
        tflops = flops / (elapsed_time * 10**12)
        
        return tflops
    
    finally:
        # Clean up tensors to free memory
        del Q, K, V, attn
        clean_up_memory()

def benchmark_layernorm(B, N, dtype, device, num_iterations, num_warmup_iterations):
    """Benchmark LayerNorm operation"""
    start = arch.event(enable_timing=True)
    end = arch.event(enable_timing=True)
    
    try:
        # Adjust dimensions
        seq_len = min(N, 2048)
        hidden_dim = min(N, 1024)
        
        # Create input tensor (B, seq_len, hidden_dim)
        x = torch.randn(B, seq_len, hidden_dim, dtype=dtype, device=device)
        
        # Create layer normalization
        ln = torch.nn.LayerNorm(hidden_dim, device=device).to(dtype=dtype)
        
        times = np.zeros(num_iterations + num_warmup_iterations)
        
        for i in range(num_warmup_iterations + num_iterations):
            arch.clear_cache()
            with torch.no_grad():
                start.record()
                ln(x)
                end.record()
            arch.synchronize()
            times[i] = start.elapsed_time(end)
        
        times = times[num_warmup_iterations:]
        elapsed_time = np.median(times) / 1000  # seconds
        
        # Approximate FLOPs for LayerNorm
        # Mean + variance + normalize = ~8 ops per element
        flops = 8 * B * seq_len * hidden_dim
        tflops = flops / (elapsed_time * 10**12)
        
        return tflops
    
    finally:
        # Clean up tensors to free memory
        del x, ln
        clean_up_memory()

def benchmark_softmax(B, N, dtype, device, num_iterations, num_warmup_iterations):
    """Benchmark Softmax operation"""
    start = arch.event(enable_timing=True)
    end = arch.event(enable_timing=True)
    
    try:
        # Create input tensor (B, N, N) like attention scores
        x = torch.randn(B, N, N, dtype=dtype, device=device)
        
        # Create softmax function
        sm = torch.nn.Softmax(dim=-1)
        
        times = np.zeros(num_iterations + num_warmup_iterations)
        
        for i in range(num_warmup_iterations + num_iterations):
            arch.clear_cache()
            with torch.no_grad():
                start.record()
                sm(x)
                end.record()
            arch.synchronize()
            times[i] = start.elapsed_time(end)
        
        times = times[num_warmup_iterations:]
        elapsed_time = np.median(times) / 1000  # seconds
        
        # Approximate FLOPs for Softmax
        # ~5 ops per element (exp, sum, divide)
        flops = 5 * B * N * N
        tflops = flops / (elapsed_time * 10**12)
        
        return tflops
    
    finally:
        # Clean up tensors to free memory
        del x
        clean_up_memory()

def run_tensor_benchmark(args, op_type, dtype_str, B, N):
    """Run a specific tensor benchmark configuration"""
    set_trial_status(True)  # Mark that a trial is starting
    
    dtype = get_dtype(dtype_str)
    device = arch.device()
    
    # Check if operation is feasible with current memory
    if args.memory_safety:
        is_feasible, est_memory = check_memory_feasibility(B, N, op_type, dtype_str, device)
        if not is_feasible:
            # Skip this configuration
            mem_gb = est_memory / (1024**3)
            print(f"\nSkipping {op_type}, {dtype_str}, B={B}, N={N} (estimated {mem_gb:.2f} GB required)")
            return None
    
    try:
        # Force clean up before running a new test
        clean_up_memory()
        
        if op_type == "matmul":
            tflops = benchmark_matmul(B, N, dtype, device, args.num_iterations, args.num_warmup_iterations)
        elif op_type == "conv2d":
            tflops = benchmark_conv2d(B, N, dtype, device, args.num_iterations, args.num_warmup_iterations)
        elif op_type == "attention":
            tflops = benchmark_attention(B, N, dtype, device, args.num_iterations, args.num_warmup_iterations)
        elif op_type == "layernorm":
            tflops = benchmark_layernorm(B, N, dtype, device, args.num_iterations, args.num_warmup_iterations)
        elif op_type == "softmax":
            tflops = benchmark_softmax(B, N, dtype, device, args.num_iterations, args.num_warmup_iterations)
        else:
            raise ValueError(f"Unknown operation type: {op_type}")
        
        return tflops
    
    except torch.cuda.OutOfMemoryError:
        print(f"\nOut of memory for {op_type}, {dtype_str}, B={B}, N={N}")
        clean_up_memory()
        return None
    
    except Exception as e:
        print(f"\nError with {op_type}, {dtype_str}, B={B}, N={N}: {e}")
        clean_up_memory()
        return None
    
    finally:
        set_trial_status(False)  # Mark that the trial has completed

def objective(trial, args):
    """Objective function for Optuna optimization"""
    # Select operation to optimize
    if "all" in args.operations:
        operations = ["matmul", "conv2d", "attention", "layernorm", "softmax"]
        op_type = trial.suggest_categorical("operation", operations)
    else:
        op_type = trial.suggest_categorical("operation", args.operations)
    
    # Select data type
    dtype_str = trial.suggest_categorical("dtype", args.dtypes)
    
    # Select batch size and tensor size
    B = trial.suggest_categorical("batch_size", args.batch_sizes)
    N = trial.suggest_categorical("tensor_size", args.tensor_sizes)
    
    # Check if operation is feasible with current memory
    if args.memory_safety:
        device = arch.device()
        is_feasible, _ = check_memory_feasibility(B, N, op_type, dtype_str, device)
        if not is_feasible:
            # Skip this configuration with a pruning mechanism
            raise optuna.exceptions.TrialPruned(f"Configuration would exceed memory")
    
    # Run the benchmark
    tflops = run_tensor_benchmark(args, op_type, dtype_str, B, N)
    if tflops is None:
        raise optuna.exceptions.TrialPruned("Failed to run benchmark")
    
    return tflops

def print_progress(study, trial):
    """Print progress during optimization"""
    print(
        f"Trial {trial.number:>4} | {trial.value:>7.2f} TFLOPS | "
        f"{trial.params['operation']:<9} | {trial.params['dtype']:<8} | "
        f"B={trial.params['batch_size']:<4} | N={trial.params['tensor_size']:<5} | "
        f"Best: {study.best_value:>7.2f} TFLOPS",
        end="\r",
    )

def save_results_to_db(db_path, benchmark_results, config_info):
    """Save benchmark results to SQLite database instead of CSV"""
    import sqlite3
    import json
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tensor_benchmark_results table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tensor_benchmark_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        operation TEXT NOT NULL,
        dtype TEXT NOT NULL,
        batch_size INTEGER NOT NULL,
        tensor_size INTEGER NOT NULL,
        tflops REAL NOT NULL,
        config_info TEXT
    )
    ''')
    
    # Save each benchmark result
    timestamp = datetime.datetime.now().isoformat()
    
    for result in benchmark_results:
        cursor.execute('''
        INSERT INTO tensor_benchmark_results
        (timestamp, operation, dtype, batch_size, tensor_size, tflops, config_info)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp,
            result['operation'],
            result['dtype'],
            result['batch_size'],
            result['tensor_size'],
            result['tflops'],
            json.dumps(config_info)
        ))
    
    conn.commit()
    conn.close()
    
    # Also use the db_utils version for uniformity
    perf_data = {
        "avg_tflops": sum(result['tflops'] for result in benchmark_results) / len(benchmark_results) if benchmark_results else 0,
        "max_tflops": max(result['tflops'] for result in benchmark_results) if benchmark_results else 0,
        "total_configs_tested": len(benchmark_results)
    }
    
    save_benchmark_result(
        db_path, 
        'tensor', 
        perf_data, 
        config_info
    )

def read_results_from_db(db_path, limit=None, operation=None, dtype=None, min_tflops=None):
    """
    Query benchmark results from the database with optional filters
    
    Args:
        db_path: Path to the SQLite database
        limit: Maximum number of results to retrieve (None for all)
        operation: Filter by specific operation (e.g. "matmul")
        dtype: Filter by specific dtype (e.g. "float16") 
        min_tflops: Filter by minimum TFLOPS value
        
    Returns:
        pandas DataFrame with results
    """
    import sqlite3
    import pandas as pd
    
    conn = sqlite3.connect(db_path)
    
    # Build query with filters
    query = "SELECT * FROM tensor_benchmark_results"
    filters = []
    params = []
    
    if operation:
        filters.append("operation = ?")
        params.append(operation)
    
    if dtype:
        filters.append("dtype = ?")
        params.append(dtype)
        
    if min_tflops:
        filters.append("tflops >= ?")
        params.append(min_tflops)
        
    if filters:
        query += " WHERE " + " AND ".join(filters)
        
    # Add ordering
    query += " ORDER BY tflops DESC"
    
    # Add limit if specified
    if limit:
        query += " LIMIT ?"
        params.append(limit)
    
    # Execute query
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    return df

def generate_db_report(db_path, output_dir=None):
    """Generate analysis report from database results"""
    import sqlite3
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import json
    from pathlib import Path
    
    # Set default output directory if not specified
    if output_dir is None:
        output_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent / "results" / "reports" / f"tensor_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_dir = Path(output_dir)
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all results
    df = read_results_from_db(db_path)
    if len(df) == 0:
        print("No benchmark results found in database")
        return
    
    # Basic statistics
    stats = {
        "total_benchmarks": len(df),
        "operations": df['operation'].unique().tolist(),
        "dtypes": df['dtype'].unique().tolist(),
        "batch_sizes": sorted(df['batch_size'].unique().tolist()),
        "tensor_sizes": sorted(df['tensor_size'].unique().tolist()),
        "max_tflops": df['tflops'].max(),
        "min_tflops": df['tflops'].min(),
        "avg_tflops": df['tflops'].mean(),
        "best_config": df.loc[df['tflops'].idxmax()].to_dict()
    }
    
    # Save stats to JSON
    with open(os.path.join(output_dir, "statistics.json"), 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    # Generate plots
    
    # 1. Overall best configurations
    plt.figure(figsize=(12, 8))
    top_n = min(10, len(df))
    top_df = df.nlargest(top_n, 'tflops')
    labels = [f"{row['operation']}, {row['dtype']}, B={row['batch_size']}, N={row['tensor_size']}" 
              for _, row in top_df.iterrows()]
    plt.barh(range(len(labels)), top_df['tflops'], color='skyblue')
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('TFLOPS')
    plt.title('Top Tensor Benchmark Configurations')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_configs.png"))
    
    # 2. Best performance by operation
    plt.figure(figsize=(10, 6))
    op_max = df.groupby('operation')['tflops'].max().sort_values(ascending=False)
    op_max.plot(kind='bar', color='skyblue')
    plt.title('Maximum TFLOPS by Operation')
    plt.ylabel('TFLOPS')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "max_by_operation.png"))
    
    # 3. Performance comparison by dtype for each operation
    for op in df['operation'].unique():
        op_df = df[df['operation'] == op]
        plt.figure(figsize=(10, 6))
        
        dtype_perf = op_df.groupby(['dtype', 'tensor_size'])['tflops'].max().unstack()
        if not dtype_perf.empty:
            dtype_perf.plot(marker='o')
            plt.title(f'{op.upper()}: Performance by Data Type and Tensor Size')
            plt.xlabel('Tensor Size')
            plt.ylabel('TFLOPS')
            plt.grid(True)
            plt.legend(title='Data Type')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{op}_dtype_comparison.png"))
    
    # 4. Heatmaps for each operation (batch_size vs tensor_size)
    for op in df['operation'].unique():
        # Filter for this operation and get the best dtype
        op_df = df[df['operation'] == op]
        best_dtype = op_df.loc[op_df['tflops'].idxmax()]['dtype']
        
        # Create heatmap for best dtype
        dtype_df = op_df[op_df['dtype'] == best_dtype]
        
        if len(dtype_df) > 0:
            # Create pivot table
            pivot_df = dtype_df.pivot_table(
                index='batch_size', 
                columns='tensor_size', 
                values='tflops',
                aggfunc='max'
            )
            
            if not pivot_df.empty:
                plt.figure(figsize=(12, 8))
                ax = plt.subplot()
                im = ax.imshow(pivot_df.values, cmap='YlGnBu')
                
                # Labeling
                ax.set_xticks(np.arange(len(pivot_df.columns)))
                ax.set_yticks(np.arange(len(pivot_df.index)))
                ax.set_xticklabels(pivot_df.columns)
                ax.set_yticklabels(pivot_df.index)
                
                # Add colorbar
                cbar = plt.colorbar(im)
                cbar.set_label('TFLOPS')
                
                # Add values to cells
                for i in range(len(pivot_df.index)):
                    for j in range(len(pivot_df.columns)):
                        if not np.isnan(pivot_df.values[i, j]):
                            value = pivot_df.values[i, j]
                            text_color = "black" if value < pivot_df.values.max() / 2 else "white"
                            ax.text(j, i, f"{value:.1f}", ha="center", va="center", color=text_color)
                
                plt.title(f"{op.upper()}: Performance with {best_dtype}")
                plt.xlabel('Tensor Size')
                plt.ylabel('Batch Size')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{op}_heatmap.png"))
    
    # Print report summary
    print(f"\nAnalysis report generated at: {output_dir}")
    print(f"Total benchmarks analyzed: {stats['total_benchmarks']}")
    print(f"Best configuration: {stats['best_config']['operation']}, {stats['best_config']['dtype']}, "
          f"B={stats['best_config']['batch_size']}, N={stats['best_config']['tensor_size']} "
          f"({stats['best_config']['tflops']:.2f} TFLOPS)")
          
    return output_dir

def visualize_optuna_results(study, output_dir):
    """Create visualizations of the Optuna study and save them to output_dir"""
    try:
        import optuna.visualization as vis
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Optimization history plot
        fig_history = vis.plot_optimization_history(study)
        fig_history.write_image(os.path.join(output_dir, "optimization_history.png"))
        
        # 2. Parameter importance
        fig_importance = vis.plot_param_importances(study)
        fig_importance.write_image(os.path.join(output_dir, "param_importance.png"))
        
        # 3. Parallel coordinate plot
        fig_parallel = vis.plot_parallel_coordinate(study, params=["operation", "dtype", "batch_size", "tensor_size"])
        fig_parallel.write_image(os.path.join(output_dir, "parallel_coordinate.png"))
        
        # 4. Contour plot for batch_size vs tensor_size
        fig_contour = vis.plot_contour(study, params=["batch_size", "tensor_size"])
        fig_contour.write_image(os.path.join(output_dir, "contour_plot.png"))
        
        # 5. Slice plot
        fig_slice = vis.plot_slice(study)
        fig_slice.write_image(os.path.join(output_dir, "slice_plot.png"))
        
        # Create custom plots with matplotlib
        
        # 6. Performance by operation type
        operations = set(trial.params['operation'] for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE)
        plt.figure(figsize=(10, 6))
        for op in operations:
            op_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE and trial.params['operation'] == op]
            if op_trials:
                values = [trial.value for trial in op_trials]
                plt.plot(values, label=op)
        plt.xlabel('Trial number within operation')
        plt.ylabel('TFLOPS')
        plt.title('Performance by Operation Type')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "performance_by_operation.png"))
        
        # 7. Performance by dtype
        dtypes = set(trial.params['dtype'] for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE)
        plt.figure(figsize=(10, 6))
        for dt in dtypes:
            dt_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE and trial.params['dtype'] == dt]
            if dt_trials:
                values = [trial.value for trial in dt_trials]
                plt.plot(values, label=dt)
        plt.xlabel('Trial number within dtype')
        plt.ylabel('TFLOPS')
        plt.title('Performance by Data Type')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "performance_by_dtype.png"))
        
        # 8. Heatmap of best performance by operation x dtype
        op_dt_perf = {}
        for op in operations:
            op_dt_perf[op] = {}
            for dt in dtypes:
                # Get trials with this operation and dtype
                filtered_trials = [t for t in study.trials if 
                                t.state == optuna.trial.TrialState.COMPLETE and 
                                t.params['operation'] == op and 
                                t.params['dtype'] == dt]
                if filtered_trials:
                    # Find best value
                    best_val = max(filtered_trials, key=lambda t: t.value).value
                    op_dt_perf[op][dt] = best_val
                else:
                    op_dt_perf[op][dt] = 0
        
        # Create DataFrame for heatmap
        heatmap_data = pd.DataFrame.from_dict(op_dt_perf, orient='index')
        
        plt.figure(figsize=(10, 8))
        ax = plt.subplot()
        im = ax.imshow(heatmap_data.values, cmap='YlGnBu')
        
        # Labeling
        ax.set_xticks(np.arange(len(dtypes)))
        ax.set_yticks(np.arange(len(operations)))
        ax.set_xticklabels(heatmap_data.columns)
        ax.set_yticklabels(heatmap_data.index)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('TFLOPS')
        
        # Add values to cells
        for i in range(len(operations)):
            for j in range(len(dtypes)):
                value = heatmap_data.iloc[i, j]
                text_color = "black" if value < heatmap_data.values.max() / 2 else "white"
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", color=text_color)
        
        plt.title("Best Performance (TFLOPS) by Operation and Data Type")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "performance_heatmap.png"))
        
        print(f"\nVisualization results saved to: {output_dir}")
        
    except ImportError as e:
        print(f"\nCould not create visualizations. Missing dependency: {e}")
    except Exception as e:
        print(f"\nError creating visualizations: {e}")

def run_benchmark(args):
    """Main entry point for running the tensor operations benchmark"""
    # Check if we should analyze existing results instead of running benchmarks
    if args.analyze_db:
        print("\n" + "="*80)
        print(f"Analyzing Tensor Operations Benchmark Results")
        print("="*80)
        
        # Use the db_file from args
        db_path = args.db_file
        
        # Ensure the database exists
        if not os.path.exists(db_path):
            print(f"Error: Database file not found: {db_path}")
            return 0, "database not found"
            
        print(f"Loading benchmark results from: {db_path}")
        
        # Generate analysis report
        report_dir = generate_db_report(
            db_path, 
            output_dir=args.report_dir
        )
        
        # Return a placeholder value
        return 0, f"analysis report generated at {report_dir}"
    
    # Continue with the existing benchmark code
    device = arch.device()
    
    # Create results directory if it doesn't exist
    results_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent / "results"
    results_dir.mkdir(exist_ok=True)
    reports_dir = results_dir / "reports" / f"tensor_ops_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("\n" + "="*80)
    print(f"Starting Tensor Operations Benchmark on {device}")
    print("="*80)
    
    # Check for tensor cores
    has_tensor_cores = False
    if arch.name() == "cuda":
        device_info = arch.device_info()
        cuda_capability = f"{device_info.major}.{device_info.minor}"
        # Tensor cores available in Volta (7.0+), Turing (7.5+), Ampere (8.0+), Ada Lovelace (8.9+), and Hopper (9.0+)
        if device_info.major >= 7:
            has_tensor_cores = True
            print(f"GPU with CUDA Capability {cuda_capability} - Tensor Cores are available")
        else:
            print(f"GPU with CUDA Capability {cuda_capability} - No Tensor Cores available")
            
    # Print memory safety settings
    print(f"Memory Safety Mode: {'Enabled' if args.memory_safety else 'Disabled'}")
    
    # Print GPU memory information
    total_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    reserved_memory_gb = torch.cuda.memory_reserved(device) / (1024**3)
    allocated_memory_gb = torch.cuda.memory_allocated(device) / (1024**3)
    available_memory_gb = total_memory_gb - reserved_memory_gb
    
    print(f"GPU Memory: {total_memory_gb:.1f} GB total, {available_memory_gb:.1f} GB available")
    
    # Print benchmark configurations
    print(f"\nBenchmark Configuration:")
    print(f"  Operations:    {args.operations}")
    print(f"  Data Types:    {args.dtypes}")
    print(f"  Batch Sizes:   {args.batch_sizes}")
    print(f"  Tensor Sizes:  {args.tensor_sizes}")
    print(f"  Iterations:    {args.num_iterations} (+ {args.num_warmup_iterations} warmup)")
    
    start_time = time.time()
    all_results = []
    best_overall = {"tflops": 0, "config": ""}
    
    if args.optimize_config:
        # Use the db_file from args - it now has the full path from main.py
        db_path = args.db_file
        
        # Create a SQLite storage with the benchmark-specific database
        storage = RDBStorage(
            url=f"sqlite:///{db_path}", 
            engine_kwargs={"connect_args": {"timeout": 30}}
        )

        print(f"\nRunning optimization to find best tensor configurations...")
        print(f"Database: {db_path}")
        
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        study.set_metric_names(["TFLOPS"])
        
        # Run optimization
        study.optimize(
            lambda trial: objective(trial, args),
            n_trials=args.n_trials,
            callbacks=[print_progress],
            catch=(RuntimeError, torch.cuda.OutOfMemoryError)  # Continue on OOM errors
        )
        
        # Get best result
        best_trial = study.best_trial
        best_overall = {
            "tflops": best_trial.value,
            "config": (
                f"{best_trial.params['operation']} with "
                f"{best_trial.params['dtype']} at "
                f"B={best_trial.params['batch_size']}, "
                f"N={best_trial.params['tensor_size']}"
            )
        }
        
        # Print best result
        print("\n\nOptimization Complete!")
        print(f"Best Configuration: {best_overall['config']}")
        print(f"Performance: {best_overall['tflops']:.2f} TFLOPS")
        
        # Print top 5 results per operation
        print("\nTop 5 Configurations per Operation:")
        if "all" in args.operations:
            ops_to_check = ["matmul", "conv2d", "attention", "layernorm", "softmax"]
        else:
            ops_to_check = args.operations
            
        for op in ops_to_check:
            # Filter trials for this operation
            op_trials = [t for t in study.trials if 
                        t.state == optuna.trial.TrialState.COMPLETE and 
                        t.params.get('operation') == op]
            if not op_trials:
                continue
                
            # Sort by value (descending)
            op_trials.sort(key=lambda t: t.value, reverse=True)
            
            print(f"\n{op.upper()}:")
            for i, trial in enumerate(op_trials[:5]):
                print(f"  {i+1}. {trial.value:.2f} TFLOPS - "
                      f"{trial.params['dtype']}, "
                      f"B={trial.params['batch_size']}, "
                      f"N={trial.params['tensor_size']}")
        
        # Create visualizations
        visualize_optuna_results(study, reports_dir)
        
    else:
        # Run fixed benchmark for all combinations
        print("\nRunning fixed benchmarks for all configurations...")
        
        # Determine which operations to run
        operations = []
        if "all" in args.operations:
            operations = ["matmul", "conv2d", "attention", "layernorm", "softmax"]
        else:
            operations = args.operations
            
        # Track best configurations for each operation
        best_per_op = {op: {"tflops": 0, "config": ""} for op in operations}
        
        # Calculate total number of configurations
        total_configs = len(operations) * len(args.dtypes) * len(args.batch_sizes) * len(args.tensor_sizes)
        print(f"Total configurations: {total_configs}")
        
        # Run all benchmarks
        config_count = 0
        successful_configs = 0
        
        for op_type in operations:
            for dtype_str in args.dtypes:
                for B in args.batch_sizes:
                    for N in args.tensor_sizes:
                        config_count += 1
                        
                        # Print progress
                        progress = f"[{config_count}/{total_configs}]"
                        config = f"{op_type}, {dtype_str}, B={B}, N={N}"
                        print(f"{progress} Testing {config}...", end="\r")
                        
                        # Check if termination was requested before running this trial
                        if should_terminate():
                            print("\n\033[1;33mBenchmark stopped gracefully as requested\033[0m")
                            return best_overall["tflops"], best_overall["config"]
                        
                        # Run benchmark
                        try:
                            tflops = run_tensor_benchmark(args, op_type, dtype_str, B, N)
                            
                            if tflops is not None:
                                successful_configs += 1
                                
                                # Store result
                                result = {
                                    "operation": op_type,
                                    "dtype": dtype_str,
                                    "batch_size": B,
                                    "tensor_size": N,
                                    "tflops": tflops
                                }
                                all_results.append(result)
                                
                                # Update best result for this operation
                                if tflops > best_per_op[op_type]["tflops"]:
                                    best_per_op[op_type] = {
                                        "tflops": tflops,
                                        "config": f"{dtype_str}, B={B}, N={N}"
                                    }
                                    
                                # Update best overall result
                                if tflops > best_overall["tflops"]:
                                    best_overall = {
                                        "tflops": tflops,
                                        "config": f"{op_type}, {dtype_str}, B={B}, N={N}"
                                    }
                        except Exception as e:
                            print(f"\nError with {config}: {e}")
                        
                        # Check if termination was requested after each trial
                        if TERMINATE_REQUESTED:
                            print("\n\033[1;33mStopping benchmark as requested\033[0m")
                            return best_overall["tflops"], best_overall["config"]
        
        # Print best results
        print("\n\nBenchmark Complete!")
        print(f"Successful configurations: {successful_configs}/{total_configs}")
        
        if successful_configs > 0:
            print(f"Best Overall: {best_overall['tflops']:.2f} TFLOPS - {best_overall['config']}")
            
            print("\nBest Configuration per Operation:")
            for op, best in best_per_op.items():
                if best['tflops'] > 0:
                    print(f"  {op.upper()}: {best['tflops']:.2f} TFLOPS - {best['config']}")
                
            # Save results to database  
            config_info = {
                "operations": args.operations,
                "dtypes": args.dtypes,
                "batch_sizes": args.batch_sizes,
                "tensor_sizes": args.tensor_sizes,
                "iterations": args.num_iterations,
                "warmup_iterations": args.num_warmup_iterations,
                "device": str(device),
                "tensor_cores_available": has_tensor_cores,
                "memory_safety": args.memory_safety
            }
            
            # Format the data for the unified db_utils and summary
            perf_data = {
                "tflops": best_overall["tflops"],
                "avg_tflops": sum(r['tflops'] for r in all_results) / len(all_results) if all_results else 0,
                "successful_configs": successful_configs,
                "total_configs": total_configs,
            }
            
            for op in best_per_op:
                if best_per_op[op]["tflops"] > 0:
                    perf_data[f"{op}_tflops"] = best_per_op[op]["tflops"]
            
            # Add best configuration to the config
            config_info["best_config"] = best_overall["config"]
            
            try:
                # Use the unified save_benchmark_result
                benchmark_id = save_benchmark_result(
                    args.db_file, 
                    'tensor', 
                    perf_data, 
                    config_info
                )
                print(f"\nResults saved to database (ID: {benchmark_id})")
                
                # Also save to custom format
                save_results_to_db(args.db_file, all_results, config_info)
                
                # Update benchmark summary file if available
                if hasattr(args, 'summary_file') and args.summary_file:
                    device_info = str(arch.device_info())
                    if update_benchmark_summary(args.summary_file, 'tensor', perf_data, config_info, device_info):
                        print(f"Benchmark result added to summary: {args.summary_file}")
                        args.summary_updated = True
            
            except Exception as e:
                print(f"\nError saving to database: {e}")
                # Fallback to CSV if database save fails
                # Save as CSV instead
                results_df = pd.DataFrame(all_results)
                csv_path = results_dir / f"tensor_ops_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                results_df.to_csv(csv_path, index=False)
                print(f"\nFallback: Results saved to CSV: {csv_path}")
            
            # Create simple visualizations
            os.makedirs(reports_dir, exist_ok=True)
            
            try:
                plt.figure(figsize=(12, 8))
                
                # Group by operation
                op_groups = results_df.groupby('operation')
                for op_name, op_data in op_groups:
                    # For each operation, get best performance for each tensor size
                    pivot = op_data.pivot_table(
                        index='tensor_size', 
                        columns='batch_size', 
                        values='tflops', 
                        aggfunc='max'
                    )
                    
                    # Plot best performance vs tensor size for different batch sizes
                    plt.figure(figsize=(10, 6))
                    pivot.plot(marker='o')
                    plt.title(f'{op_name.upper()}: TFLOPS vs. Tensor Size')
                    plt.xlabel('Tensor Size (N)')
                    plt.ylabel('TFLOPS')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(reports_dir, f"{op_name}_performance.png"))
                
                # Create heatmaps for each operation
                for op_name, op_data in op_groups:
                    # Create pivot table
                    heatmap_data = op_data.pivot_table(
                        index='batch_size', 
                        columns='tensor_size', 
                        values='tflops',
                        aggfunc='max'
                    )
                    
                    plt.figure(figsize=(10, 8))
                    ax = plt.subplot()
                    im = ax.imshow(heatmap_data.values, cmap='YlGnBu')
                    
                    # Labeling
                    ax.set_xticks(np.arange(len(heatmap_data.columns)))
                    ax.set_yticks(np.arange(len(heatmap_data.index)))
                    ax.set_xticklabels(heatmap_data.columns)
                    ax.set_yticklabels(heatmap_data.index)
                    
                    # Add colorbar
                    cbar = plt.colorbar(im)
                    cbar.set_label('TFLOPS')
                    
                    # Add values to cells
                    for i in range(len(heatmap_data.index)):
                        for j in range(len(heatmap_data.columns)):
                            if not np.isnan(heatmap_data.values[i, j]):
                                value = heatmap_data.values[i, j]
                                text_color = "black" if value < heatmap_data.values.max() / 2 else "white"
                                ax.text(j, i, f"{value:.1f}", ha="center", va="center", color=text_color)
                    
                    plt.title(f"{op_name.upper()}: Performance by Batch Size and Tensor Size")
                    plt.xlabel('Tensor Size (N)')
                    plt.ylabel('Batch Size (B)')
                    plt.tight_layout()
                    plt.savefig(os.path.join(reports_dir, f"{op_name}_heatmap.png"))
                
                # Create overall comparison of operations
                plt.figure(figsize=(12, 8))
                op_perf = results_df.groupby('operation')['tflops'].max()
                op_perf.plot(kind='bar', color='skyblue')
                plt.title('Maximum Performance by Operation')
                plt.xlabel('Operation')
                plt.ylabel('TFLOPS')
                plt.grid(axis='y')
                plt.tight_layout()
                plt.savefig(os.path.join(reports_dir, "operations_comparison.png"))
                
                # Performance by dtype
                plt.figure(figsize=(10, 6))
                dtype_perf = results_df.groupby(['operation', 'dtype'])['tflops'].max().unstack()
                dtype_perf.plot(kind='bar')
                plt.title('Maximum Performance by Operation and Data Type')
                plt.xlabel('Operation')
                plt.ylabel('TFLOPS')
                plt.grid(axis='y')
                plt.tight_layout()
                plt.savefig(os.path.join(reports_dir, "dtype_comparison.png"))
                
                print(f"\nVisualization reports saved to: {reports_dir}")
                
            except Exception as e:
                print(f"\nError creating visualizations: {e}")
                
        else:
            print("No successful benchmark configurations were completed.")
    
    # Print elapsed time
    time_delta = time.time() - start_time
    time_str = str(datetime.timedelta(seconds=time_delta)).split(".")[0]
    print(f"\nTotal benchmark time: {time_str}")
    
    return best_overall["tflops"], best_overall["config"]