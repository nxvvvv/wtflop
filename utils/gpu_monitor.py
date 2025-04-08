import sqlite3
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

class GPUMonitor:
    """
    Monitor NVIDIA GPU metrics using nvidia-smi and store results in a SQLite database
    """
    def __init__(self, db_path, interval=1.0):
        """
        Initialize the GPU monitor
        
        Args:
            db_path: Path to save the SQLite database
            interval: Sampling interval in seconds
        """
        self.db_path = db_path
        self.interval = interval
        self.stop_event = threading.Event()
        self.monitor_thread = None
        self.benchmark_id = None
        self.benchmark_type = None
        self._setup_db()
        
    def _setup_db(self):
        """Create the database schema if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
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
        
        # Check if we need to create benchmark-specific metrics tables
        benchmark_types = ["mamf", "tensor"]
        
        for benchmark_type in benchmark_types:
            table_name = f"gpu_metrics_{benchmark_type}"
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                benchmark_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                gpu_id INTEGER NOT NULL,
                utilization_gpu INTEGER,
                utilization_memory INTEGER,
                temperature_gpu INTEGER,
                power_draw REAL,
                memory_used INTEGER,
                memory_total INTEGER,
                FOREIGN KEY (benchmark_id) REFERENCES benchmarks(id)
            )
            ''')
        
        conn.commit()
        conn.close()
    
    def _get_gpu_metrics(self):
        """
        Get GPU metrics from nvidia-smi
        
        Returns:
            List of dictionaries with GPU metrics
        """
        try:
            # Run nvidia-smi with specific format to get utilization, temp, power, memory
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,temperature.gpu,power.draw,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            metrics = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                    
                values = [val.strip() for val in line.split(',')]
                if len(values) == 7:
                    metric = {
                        'gpu_id': int(values[0]),
                        'utilization_gpu': int(values[1]),
                        'utilization_memory': int(values[2]),
                        'temperature_gpu': int(values[3]),
                        'power_draw': float(values[4]) if values[4] != 'N/A' else None,
                        'memory_used': int(values[5]),
                        'memory_total': int(values[6])
                    }
                    metrics.append(metric)
                    
            return metrics
        except (subprocess.SubprocessError, ValueError, IndexError) as e:
            print(f"Error getting GPU metrics: {e}")
            return []
    
    def _monitor_loop(self):
        """Background monitoring thread loop"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Use benchmark type-specific table
        metrics_table = f"gpu_metrics_{self.benchmark_type}"
        
        while not self.stop_event.is_set():
            try:
                metrics = self._get_gpu_metrics()
                timestamp = datetime.now().isoformat()
                
                for metric in metrics:
                    cursor.execute(f'''
                    INSERT INTO {metrics_table} (
                        benchmark_id, timestamp, gpu_id, utilization_gpu, 
                        utilization_memory, temperature_gpu, power_draw, 
                        memory_used, memory_total
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        self.benchmark_id,
                        timestamp,
                        metric['gpu_id'],
                        metric['utilization_gpu'],
                        metric['utilization_memory'],
                        metric['temperature_gpu'],
                        metric['power_draw'],
                        metric['memory_used'],
                        metric['memory_total']
                    ))
                
                conn.commit()
            except Exception as e:
                print(f"Error in GPU monitoring: {e}")
            
            # Sleep for interval duration
            time.sleep(self.interval)
        
        conn.close()
    
    def start_monitoring(self, benchmark_name, benchmark_type, parameters=None):
        """
        Start GPU monitoring for a benchmark
        
        Args:
            benchmark_name: Name of the benchmark
            benchmark_type: Type of benchmark (mamf, tensor, etc.)
            parameters: Optional dict or string with benchmark parameters
        
        Returns:
            The benchmark ID
        """
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_monitoring()
        
        # Save benchmark type
        self.benchmark_type = benchmark_type.lower()
        
        # Reset stop event
        self.stop_event.clear()
        
        # Create a new benchmark entry
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert parameters to string if needed
        if parameters and not isinstance(parameters, str):
            import json
            parameters = json.dumps(parameters)
        
        cursor.execute('''
        INSERT INTO benchmarks (name, benchmark_type, start_time, parameters)
        VALUES (?, ?, ?, ?)
        ''', (benchmark_name, self.benchmark_type, datetime.now().isoformat(), parameters))
        
        self.benchmark_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        return self.benchmark_id
    
    def stop_monitoring(self):
        """Stop GPU monitoring and update benchmark end time"""
        if not self.monitor_thread or not self.monitor_thread.is_alive():
            return
        
        # Set stop event to end monitoring thread
        self.stop_event.set()
        self.monitor_thread.join(timeout=2.0)
        
        # Update benchmark with end time
        if self.benchmark_id:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            UPDATE benchmarks
            SET end_time = ?
            WHERE id = ?
            ''', (datetime.now().isoformat(), self.benchmark_id))
            
            conn.commit()
            conn.close()