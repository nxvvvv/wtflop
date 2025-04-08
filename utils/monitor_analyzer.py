import argparse
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def get_benchmarks(db_path):
    """Get list of all benchmarks in the database"""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT id, name, benchmark_type, start_time, end_time, parameters FROM benchmarks", conn)
    conn.close()
    return df

def get_metrics(db_path, benchmark_id, benchmark_type):
    """Get all metrics for a specific benchmark"""
    conn = sqlite3.connect(db_path)
    
    # Use the appropriate metrics table based on benchmark type
    table_name = f"gpu_metrics_{benchmark_type}"
    
    query = f"""
    SELECT * FROM {table_name} 
    WHERE benchmark_id = {benchmark_id}
    ORDER BY timestamp
    """
    
    try:
        df = pd.read_sql(query, conn)
    except Exception as e:
        print(f"Error querying {table_name}: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    
    # Convert timestamp to datetime
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate elapsed time in seconds from start
        start_time = df['timestamp'].min()
        df['elapsed_seconds'] = (df['timestamp'] - start_time).dt.total_seconds()
    
    return df

def generate_report(db_path, benchmark_id, output_dir=None):
    """Generate a report for the specified benchmark"""
    conn = sqlite3.connect(db_path)
    benchmark = pd.read_sql(f"SELECT * FROM benchmarks WHERE id = {benchmark_id}", conn).iloc[0]
    conn.close()
    
    # Get benchmark type
    benchmark_type = benchmark["benchmark_type"]
    
    # Get metrics data
    metrics_df = get_metrics(db_path, benchmark_id, benchmark_type)
    
    if metrics_df.empty:
        print(f"No metrics found for benchmark {benchmark_id} (type: {benchmark_type})")
        return
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path(db_path).parent / "reports" / f"benchmark_{benchmark_id}"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary
    summary = {
        "benchmark_id": benchmark_id,
        "name": benchmark["name"],
        "benchmark_type": benchmark_type,
        "start_time": benchmark["start_time"],
        "end_time": benchmark["end_time"],
        "duration_seconds": (pd.to_datetime(benchmark["end_time"]) - 
                            pd.to_datetime(benchmark["start_time"])).total_seconds(),
        "gpu_count": metrics_df["gpu_id"].nunique(),
        "sample_count": len(metrics_df),
    }
    
    # Group by GPU to get stats
    gpu_stats = metrics_df.groupby("gpu_id").agg({
        "utilization_gpu": ["mean", "max", "min", "std"],
        "temperature_gpu": ["mean", "max", "min"],
        "power_draw": ["mean", "max", "min"],
        "memory_used": ["mean", "max", "min"]
    }).reset_index()
    
    # Generate plots
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    for gpu_id in metrics_df["gpu_id"].unique():
        gpu_data = metrics_df[metrics_df["gpu_id"] == gpu_id]
        plt.plot(gpu_data["elapsed_seconds"], gpu_data["utilization_gpu"], 
                 label=f"GPU {gpu_id}")
    plt.title("GPU Utilization")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Utilization %")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    for gpu_id in metrics_df["gpu_id"].unique():
        gpu_data = metrics_df[metrics_df["gpu_id"] == gpu_id]
        plt.plot(gpu_data["elapsed_seconds"], gpu_data["temperature_gpu"], 
                 label=f"GPU {gpu_id}")
    plt.title("GPU Temperature")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    for gpu_id in metrics_df["gpu_id"].unique():
        gpu_data = metrics_df[metrics_df["gpu_id"] == gpu_id]
        plt.plot(gpu_data["elapsed_seconds"], gpu_data["power_draw"], 
                 label=f"GPU {gpu_id}")
    plt.title("Power Consumption")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    for gpu_id in metrics_df["gpu_id"].unique():
        gpu_data = metrics_df[metrics_df["gpu_id"] == gpu_id]
        # Convert to GB for better visualization
        plt.plot(gpu_data["elapsed_seconds"], gpu_data["memory_used"] / 1024, 
                 label=f"GPU {gpu_id}")
    plt.title("Memory Usage")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Memory Used (GB)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = output_dir / "gpu_metrics.png"
    plt.savefig(plot_path)
    
    # Save summary and data
    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"Benchmark Report ID: {benchmark_id}\n")
        f.write(f"Name: {summary['name']}\n")
        f.write(f"Type: {summary['benchmark_type']}\n")
        f.write(f"Start Time: {summary['start_time']}\n")
        f.write(f"End Time: {summary['end_time']}\n")
        f.write(f"Duration: {summary['duration_seconds']:.2f} seconds\n")
        f.write(f"GPU Count: {summary['gpu_count']}\n")
        f.write(f"Sample Count: {summary['sample_count']}\n\n")
        
        f.write("GPU Statistics:\n")
        for _, row in gpu_stats.iterrows():
            gpu_id = row["gpu_id"]
            f.write(f"\nGPU {gpu_id}:\n")
            f.write(f"  Utilization: avg={row[('utilization_gpu', 'mean')]:.1f}%, "
                    f"max={row[('utilization_gpu', 'max')]}%, "
                    f"min={row[('utilization_gpu', 'min')]}%\n")
            f.write(f"  Temperature: avg={row[('temperature_gpu', 'mean')]:.1f}°C, "
                    f"max={row[('temperature_gpu', 'max')]}°C, "
                    f"min={row[('temperature_gpu', 'min')]}°C\n")
            f.write(f"  Power: avg={row[('power_draw', 'mean')]:.1f}W, "
                    f"max={row[('power_draw', 'max')]}W, "
                    f"min={row[('power_draw', 'min')]}W\n")
            f.write(f"  Memory: avg={row[('memory_used', 'mean')]/1024:.2f}GB, "
                    f"max={row[('memory_used', 'max')]/1024:.2f}GB, "
                    f"min={row[('memory_used', 'min')]/1024:.2f}GB\n")
    
    # Save raw data
    metrics_df.to_csv(output_dir / "raw_metrics.csv", index=False)
    
    print(f"Report generated in {output_dir}")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Analyze GPU monitoring data")
    parser.add_argument(
        "--db_path", 
        type=str, 
        required=True,
        help="Path to the monitoring database"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List all benchmarks in the database"
    )
    parser.add_argument(
        "--benchmark_id", 
        type=int,
        help="ID of the benchmark to analyze"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        help="Directory to save the report"
    )
    
    args = parser.parse_args()
    
    if args.list:
        benchmarks = get_benchmarks(args.db_path)
        print("\nAvailable benchmarks:")
        for _, row in benchmarks.iterrows():
            duration = "N/A"
            if row["end_time"] is not None:
                start = pd.to_datetime(row["start_time"])
                end = pd.to_datetime(row["end_time"])
                duration = f"{(end - start).total_seconds():.1f}s"
                
            print(f"ID: {row['id']} | Type: {row['benchmark_type']} | Name: {row['name']} | Started: {row['start_time']} | Duration: {duration}")
    
    if args.benchmark_id:
        generate_report(args.db_path, args.benchmark_id, args.output_dir)

if __name__ == "__main__":
    main()