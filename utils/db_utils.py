"""
Database utilities for benchmark result storage and retrieval
"""

import sqlite3
import json
import datetime
import os
from pathlib import Path

# Add this global variable to track processed results
_processed_results = set()

def setup_database(db_path):
    """Initialize benchmark database with proper schema"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create benchmarks table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS benchmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        benchmark_type TEXT NOT NULL,
        start_time TIMESTAMP NOT NULL,
        end_time TIMESTAMP,
        parameters TEXT
    )
    ''')
    
    # MAMF benchmark table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS mamf_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        benchmark_id INTEGER NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        tflops REAL,
        total_time_seconds REAL,
        parameters TEXT,
        FOREIGN KEY (benchmark_id) REFERENCES benchmarks(id)
    )
    ''')
    
    # Tensor Operations benchmark table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tensor_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        benchmark_id INTEGER NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        tflops REAL,
        avg_tflops REAL,
        successful_configs INTEGER,
        total_configs INTEGER,
        parameters TEXT,
        FOREIGN KEY (benchmark_id) REFERENCES benchmarks(id)
    )
    ''')
    
    # Tensor Detailed Results table for individual configurations
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tensor_detailed_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        benchmark_id INTEGER NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        operation TEXT,
        dtype TEXT,
        batch_size INTEGER,
        tensor_size INTEGER,
        tflops REAL,
        parameters TEXT,
        FOREIGN KEY (benchmark_id) REFERENCES benchmarks(id)
    )
    ''')
    
    # Data Generation benchmark table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS datagen_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        benchmark_id INTEGER NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        total_bandwidth_GB_per_s REAL,
        time_seconds REAL,
        per_gpu_bandwidths TEXT,
        parameters TEXT,
        FOREIGN KEY (benchmark_id) REFERENCES benchmarks(id)
    )
    ''')
    
    # Transfer benchmark table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transfer_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        benchmark_id INTEGER NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        gpu_to_cpu_bandwidth_GB_per_s REAL,
        gpu_to_gpu_bandwidth_GB_per_s REAL,
        gpu_to_cpu_time_seconds REAL,
        gpu_to_gpu_time_seconds REAL,
        parameters TEXT,
        FOREIGN KEY (benchmark_id) REFERENCES benchmarks(id)
    )
    ''')
    
    # Memory Bandwidth benchmark table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS membw_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        benchmark_id INTEGER NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        gpu_bandwidth_GB_per_s REAL,
        system_bandwidth_GB_per_s REAL,
        gpu_copy_time_seconds REAL,
        system_copy_time_seconds REAL,
        parameters TEXT,
        FOREIGN KEY (benchmark_id) REFERENCES benchmarks(id)
    )
    ''')
    
    # Inference benchmark table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS inference_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        benchmark_id INTEGER NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        throughput_samples_per_sec REAL,
        avg_latency_ms REAL,
        total_time_seconds REAL,
        parameters TEXT,
        FOREIGN KEY (benchmark_id) REFERENCES benchmarks(id)
    )
    ''')
    
    # Computation benchmark table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS compute_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        benchmark_id INTEGER NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        gflops REAL,
        total_time_seconds REAL,
        epochs_completed INTEGER,
        parameters TEXT,
        FOREIGN KEY (benchmark_id) REFERENCES benchmarks(id)
    )
    ''')
    
    conn.commit()
    conn.close()

def save_benchmark_result(db_path, benchmark_type, perf_data, config):
    """Save benchmark results to database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create timestamp for this result
    timestamp = datetime.datetime.now().isoformat()
    
    # Insert benchmark metadata
    cursor.execute('''
    INSERT INTO benchmarks
    (name, benchmark_type, start_time, end_time, parameters)
    VALUES (?, ?, ?, ?, ?)
    ''', (
        f"{benchmark_type.capitalize()} Benchmark",
        benchmark_type,
        timestamp,
        timestamp,  # Same as start time since we're saving after completion
        json.dumps(config)
    ))
    
    benchmark_id = cursor.lastrowid
    
    # Insert specific benchmark data based on type
    if benchmark_type == 'mamf':
        cursor.execute('''
        INSERT INTO mamf_results
        (benchmark_id, timestamp, tflops, total_time_seconds, parameters)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            benchmark_id,
            timestamp,
            perf_data.get('tflops', 0),
            perf_data.get('total_time_seconds', 0),
            json.dumps(config)
        ))
    elif benchmark_type == 'tensor':
        cursor.execute('''
        INSERT INTO tensor_results
        (benchmark_id, timestamp, tflops, avg_tflops, successful_configs, total_configs, parameters)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            benchmark_id,
            timestamp,
            perf_data.get('tflops', 0),
            perf_data.get('avg_tflops', 0),
            perf_data.get('successful_configs', 0),
            perf_data.get('total_configs', 0),
            json.dumps(config)
        ))
    elif benchmark_type == 'datagen':
        cursor.execute('''
        INSERT INTO datagen_results
        (benchmark_id, timestamp, total_bandwidth_GB_per_s, time_seconds, per_gpu_bandwidths, parameters)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            benchmark_id,
            timestamp,
            perf_data.get('total_bandwidth_GB_per_s', 0),
            perf_data.get('time_seconds', 0),
            json.dumps(perf_data.get('per_gpu_bandwidths_GB_per_s', [])),
            json.dumps(config)
        ))
    elif benchmark_type == 'transfer':
        cursor.execute('''
        INSERT INTO transfer_results
        (benchmark_id, timestamp, gpu_to_cpu_bandwidth_GB_per_s, gpu_to_gpu_bandwidth_GB_per_s, 
        gpu_to_cpu_time_seconds, gpu_to_gpu_time_seconds, parameters)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            benchmark_id,
            timestamp,
            perf_data.get('gpu_to_cpu_bandwidth_GB_per_s', 0),
            perf_data.get('gpu_to_gpu_bandwidth_GB_per_s', 0),
            perf_data.get('gpu_to_cpu_time_seconds', 0),
            perf_data.get('gpu_to_gpu_time_seconds', 0),
            json.dumps(config)
        ))
    elif benchmark_type == 'membw':
        cursor.execute('''
        INSERT INTO membw_results
        (benchmark_id, timestamp, gpu_bandwidth_GB_per_s, system_bandwidth_GB_per_s, 
        gpu_copy_time_seconds, system_copy_time_seconds, parameters)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            benchmark_id,
            timestamp,
            perf_data.get('gpu_bandwidth_GB_per_s', 0),
            perf_data.get('system_bandwidth_GB_per_s', 0),
            perf_data.get('gpu_copy_time_seconds', 0),
            perf_data.get('system_copy_time_seconds', 0),
            json.dumps(config)
        ))
    elif benchmark_type == 'inference':
        cursor.execute('''
        INSERT INTO inference_results
        (benchmark_id, timestamp, throughput_samples_per_sec, avg_latency_ms, total_time_seconds, parameters)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            benchmark_id,
            timestamp,
            perf_data.get('throughput_samples_per_sec', 0),
            perf_data.get('avg_latency_ms', 0),
            perf_data.get('total_time_seconds', 0),
            json.dumps(config)
        ))
    elif benchmark_type == 'compute':
        cursor.execute('''
        INSERT INTO compute_results
        (benchmark_id, timestamp, gflops, total_time_seconds, epochs_completed, parameters)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            benchmark_id,
            timestamp,
            perf_data.get('gflops', 0),
            perf_data.get('total_time_seconds', 0),
            perf_data.get('epochs_completed', 0),
            json.dumps(config)
        ))
    
    conn.commit()
    conn.close()
    
    return benchmark_id

def update_benchmark_summary(*args, **kwargs):
    """Update benchmark summary with proper error handling for argument variations"""
    
    # Get the required arguments (first two)
    benchmark_name = args[0]
    result_data = args[1]
    
    # Get optional arguments if present
    summary_path = None
    skip_if_exists = False
    
    # Handle different argument patterns
    if len(args) > 2:
        # Check if the third argument is a valid path or path-like object
        if isinstance(args[2], (str, bytes, os.PathLike)) or hasattr(args[2], '__fspath__'):
            summary_path = args[2]
        else:
            # If it's not a path, it might be another parameter - ignore it
            print(f"Warning: Expected path for summary_path but got {type(args[2]).__name__}")
    
    if len(args) > 3 and isinstance(args[3], bool):
        skip_if_exists = args[3]
    
    # Get arguments from kwargs if provided
    summary_path = kwargs.get('summary_path', summary_path)
    skip_if_exists = kwargs.get('skip_if_exists', skip_if_exists)
    
    # Deduplication logic using global set
    global _processed_results
    
    # Create a hash of the result data to identify duplicates
    import hashlib
    import json
    result_str = json.dumps(result_data, sort_keys=True)
    result_hash = hashlib.md5(result_str.encode()).hexdigest()
    
    # Check if we've already processed this exact result
    result_key = f"{benchmark_name}:{result_hash}"
    if result_key in _processed_results:
        print("Skipping duplicate benchmark result update")
        return
    
    _processed_results.add(result_key)
    
    # Default path if none provided
    if summary_path is None:
        from pathlib import Path
        summary_path = Path("results") / "benchmark_summary.json"
    
    # Create directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
    
    # Load existing summary
    summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
        except json.JSONDecodeError:
            # If file is corrupted, start fresh
            summary = {}
    
    # Check if benchmark already exists
    if benchmark_name not in summary:
        summary[benchmark_name] = []
    
    # Add result to the list for this benchmark
    if isinstance(summary[benchmark_name], list):
        summary[benchmark_name].append(result_data)
    else:
        # Handle the case where it might not be a list (backward compatibility)
        summary[benchmark_name] = [result_data]
    
    # Save updated summary
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Benchmark result added to summary: {summary_path}")