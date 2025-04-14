from pathlib import Path
import argparse
import datetime
import os
import platform
import re
import shlex
import signal
import sys
import time
import json
from typing import List, Dict, Any, Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint  # Rich library's print function for formatted output
import questionary  # Library for interactive command-line interfaces
from questionary import Style
import subprocess
import threading
import select

# Fix CUDA multiprocessing issue
if platform.system() != 'Windows':  # Not needed on Windows
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Method already set
        pass

# Detect operating system for platform-specific code
IS_WINDOWS = platform.system() == 'Windows'

# Import Windows-specific keyboard handling module
if IS_WINDOWS:
    import msvcrt


# Import utility modules
from utils.arch import get_accelerator_arch  # GPU architecture detection
from utils.tee import Tee  # Output redirection to both console and file
from utils.utilities import print_benchmark_header, handle_sigkill, set_terminate
from utils.gpu_monitor import GPUMonitor  # GPU resource usage monitoring
from utils.shared_state import TERMINATE_REQUESTED, set_terminate
from utils.db_utils import setup_database, update_benchmark_summary  # Database utilities

# Set up directory structure
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = ROOT_DIR / "results"

# Create necessary directories if they don't exist
RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "logs").mkdir(exist_ok=True)  # For benchmark output logs
(RESULTS_DIR / "db").mkdir(exist_ok=True)    # For database files
(RESULTS_DIR / "reports").mkdir(exist_ok=True)  # For generated reports


# Import available benchmarks dynamically when needed
AVAILABLE_BENCHMARKS = {
    "mamf": "benchmarks.mamf",
    "tensor": "benchmarks.tensor_ops",
    "datagen": "benchmarks.data_generation",
    "transfer": "benchmarks.transfer",
    "membw": "benchmarks.memory_bandwidth",
    "inference": "benchmarks.inference",
    "compute": "benchmarks.computation"
}

def get_benchmark_function(name):
    """Dynamically import and return the benchmark function and module"""
    module_name = AVAILABLE_BENCHMARKS[name]
    module = __import__(module_name, fromlist=['run_benchmark', 'add_benchmark_args'])
    return module, module.run_benchmark


# Define custom styling for the interactive CLI
custom_style = Style([
    ('qmark', 'fg:green bold'),      # Question mark
    ('question', 'fg:white bold'),   # Question text
    ('answer', 'fg:yellow bold'),    # User's answer
    ('pointer', 'fg:cyan bold'),     # Selection pointer
    ('highlighted', 'fg:cyan'),      # Highlighted option
    ('selected', 'fg:green'),        # Selected option
    ('instruction', 'fg:white'),     # Instructions
])

# Initialize Rich console for pretty output
console = Console()

def create_parser():
    """Create the argument parser with all options"""
    parser = argparse.ArgumentParser(description="GPU Benchmarking Tool")
    # Define the benchmark to run
    parser.add_argument(
        "--benchmark", 
        type=str, 
        choices=list(AVAILABLE_BENCHMARKS.keys()),
        default="mamf",
        help="Benchmark to run"
    )
    # Output file configuration
    parser.add_argument(
        "--output-file",  # Changed to hyphen
        "--output_file",  # Keep underscore version for backward compatibility
        type=str, 
        default=None,
        help="File to save benchmark results (defaults to results/logs/<benchmark>_<timestamp>.txt)"
    )
    # Additional notes for the benchmark
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="benchmark-specific notes to add to the output file header"
    )
    # Control console output
    parser.add_argument(
        "--verbose",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="log to stdout besides output_file?"
    )
    # GPU monitoring configuration
    parser.add_argument(
        "--monitor-interval",  # Changed to hyphen
        "--monitor_interval",  # Keep underscore version for backward compatibility
        type=float,
        default=1.0,
        help="Interval in seconds between GPU monitoring samples"
    )
    parser.add_argument(
        "--monitor-db",  # Changed to hyphen
        "--monitor_db",  # Keep underscore version for backward compatibility
        type=str,
        default="gpu_monitoring.db",
        help="Database file to store GPU monitoring data"
    )
    parser.add_argument(
        "--skip-monitoring",  # Changed to hyphen
        "--skip_monitoring",  # Keep underscore version for backward compatibility
        action="store_true",
        help="Skip GPU monitoring during benchmarks"
    )
    # Fast menu option
    parser.add_argument(
        "--fast-menu",
        action="store_true",
        help="Skip GPU initialization for faster menu loading"
    )
    
    return parser

def get_benchmark_specific_options(benchmark_name):
    """Get command-line options specific to the selected benchmark"""
    parser = argparse.ArgumentParser(add_help=False)
    
    # Import the module, not just the function
    if benchmark_name == "mamf":
        import benchmarks.mamf as benchmark_module
    elif benchmark_name == "tensor":
        import benchmarks.tensor_ops as benchmark_module
    elif benchmark_name == "datagen":
        import benchmarks.data_generation as benchmark_module
    elif benchmark_name == "transfer":
        import benchmarks.transfer as benchmark_module
    elif benchmark_name == "membw":
        import benchmarks.memory_bandwidth as benchmark_module
    elif benchmark_name == "inference":
        import benchmarks.inference as benchmark_module
    elif benchmark_name == "compute":
        import benchmarks.computation as benchmark_module
    else:
        return {}
    
    # Use the module's add_benchmark_args function
    if hasattr(benchmark_module, 'add_benchmark_args'):
        benchmark_module.add_benchmark_args(parser)
    
    # Get information about arguments
    option_info = {}
    for action in parser._actions:
        if action.dest != 'help':  # Skip help action
            option_info[action.dest] = {
                'default': action.default,
                'help': action.help or f"Option {action.dest}",
                'type': action.type.__name__ if action.type else 'str',
                'choices': action.choices,
                'nargs': action.nargs
            }
    
    return option_info

def display_welcome():
    """Display welcome message and GPU information"""
    # Get GPU architecture information
    arch = get_accelerator_arch()
    device = arch.device()
    device_info = arch.device_info()
    
    # Display GPU information in a pretty panel
    rprint(Panel.fit(
        f"[bold green]GPU Benchmark Tool[/bold green]\n\n"
        f"[yellow]Device:[/yellow] {device}\n"
        f"[yellow]GPU:[/yellow] {device_info.name}\n"
        f"[yellow]CUDA Capability:[/yellow] {device_info.major}.{device_info.minor}\n"
        f"[yellow]Memory:[/yellow] {device_info.total_memory/1024/1024:.1f} GB\n",
        title="Welcome",
        border_style="blue"
    ))

def select_benchmark():
    """Allow user to select a benchmark to run"""
    choices = [
        {
            "name": "Matrix Multiplication Benchmark (MAMF)",
            "value": "mamf"
        },
        {
            "name": "Tensor Operations Benchmark",
            "value": "tensor"
        },
        {
            "name": "GPU Data Generation Benchmark",
            "value": "datagen"
        },
        {
            "name": "GPU Transfer Benchmarks (GPU-to-CPU, GPU-to-GPU)",
            "value": "transfer"
        },
        {
            "name": "Memory Bandwidth Benchmarks (GPU & System)",
            "value": "membw"
        },
        {
            "name": "GPU Inference Benchmark",
            "value": "inference"
        },
        {
            "name": "GPU Computational Benchmark",
            "value": "compute"
        }
    ]
    
    benchmark = questionary.select(
        "Select a benchmark to run:",
        choices=choices,
        style=custom_style
    ).ask()
    
    return benchmark

def configure_benchmark(benchmark_name):
    """Configure benchmark options interactively"""
    # Get available options for this benchmark
    options = get_benchmark_specific_options(benchmark_name)
    
    # Display options in a table
    table = Table(title=f"{benchmark_name.upper()} Benchmark Options")
    table.add_column("Option", style="cyan")
    table.add_column("Default", style="yellow")
    table.add_column("Description", style="green")
    
    # Add this conversion to convert simple values to option dictionaries
    formatted_options = {}
    for name, value in options.items():
        # Check if this is already a dictionary with the expected structure
        if isinstance(value, dict) and 'default' in value and 'help' in value:
            formatted_options[name] = value
        else:
            # This is a simple value from argparse, create the expected structure
            formatted_options[name] = {
                'default': value,
                'help': f"Option {name}",  # Basic description
                'type': type(value).__name__,
                'choices': None,
                'nargs': None
            }
    
    # Use the formatted options for display
    for name, details in formatted_options.items():
        default_str = str(details['default'])
        if isinstance(details['default'], list):
            default_str = str(details['default']).replace(',', ', ')
        table.add_row(name, default_str, details['help'])
    
    console.print(table)
    
    # Rest of the function using formatted_options instead of options
    # Ask if user wants to customize options
    customize = questionary.confirm(
        "Would you like to customize benchmark options?",
        default=False,
        style=custom_style
    ).ask()
    
    if not customize:
        return {}
    
    # Let user select which options to customize
    to_customize = questionary.checkbox(
        "Select options to customize:",
        choices=list(formatted_options.keys()),
        style=custom_style
    ).ask()
    
    # Configure each selected option
    custom_options = {}
    
    for option in to_customize:
        details = formatted_options[option]
        
        # Boolean options use confirm dialog
        if details['type'] == 'bool':
            value = questionary.confirm(
                f"{option} ({details['help']}):",
                default=details['default'],
                style=custom_style
            ).ask()
        
        # Options with choices use select dialog
        elif details['choices']:
            
            if isinstance(details['choices'], list):
                value = questionary.select(
                    f"{option} ({details['help']}):",
                    choices=[str(c) for c in details['choices']],
                    default=str(details['default']),
                    style=custom_style
                ).ask()
                
                # Convert to the appropriate type
                if details['type'] == 'int':
                    value = int(value)
                elif details['type'] == 'float':
                    value = float(value)
        
        # Options that accept multiple values
        elif details['nargs'] in ['+', '*'] or isinstance(details['nargs'], int):
            
            default_str = ', '.join([str(x) for x in details['default']]) if isinstance(details['default'], list) else str(details['default'])
            value_str = questionary.text(
                f"{option} ({details['help']}):\nEnter values separated by commas",
                default=default_str,
                style=custom_style
            ).ask()
            
            # Split by commas and strip whitespace
            values = [v.strip() for v in value_str.split(',')]
            
            # Convert to the appropriate type
            if details['type'] == 'int':
                value = [int(v) for v in values]
            elif details['type'] == 'float':
                value = [float(v) for v in values]
            else:
                value = values
                
        # Special case for nargs=3 (typically start, stop, step)
        elif details['nargs'] == 3:
            
            default_str = ', '.join([str(x) for x in details['default']])
            value_str = questionary.text(
                f"{option} ({details['help']}):\nEnter start, stop, step separated by commas",
                default=default_str,
                style=custom_style
            ).ask()
            
            # Split by commas and strip whitespace
            values = [v.strip() for v in value_str.split(',')]
            
            # Convert to the appropriate type
            if details['type'] == 'int':
                value = [int(v) for v in values]
            elif details['type'] == 'float':
                value = [float(v) for v in values]
            else:
                value = values
        
        # For all other option types
        else:
            
            if details['type'] == 'int':
                value = questionary.text(
                    f"{option} ({details['help']}):",
                    default=str(details['default']),
                    validate=lambda text: text.isdigit(),  # Validate integer input
                    style=custom_style
                ).ask()
                value = int(value)
                
            elif details['type'] == 'float':
                value = questionary.text(
                    f"{option} ({details['help']}):",
                    default=str(details['default']),
                    validate=lambda text: text.replace('.', '', 1).isdigit(),  # Validate float input
                    style=custom_style
                ).ask()
                value = float(value)
                
            else:
                value = questionary.text(
                    f"{option} ({details['help']}):",
                    default=str(details['default']),
                    style=custom_style
                ).ask()
        
        custom_options[option] = value
    
    return custom_options

def configure_general_options():
    """Configure general options for the benchmark"""
    general_options = {}
    
    # Ask about GPU monitoring
    monitor = questionary.confirm(
        "Enable GPU monitoring during benchmark?",
        default=True,
        style=custom_style
    ).ask()
    
    if not monitor:
        general_options["skip_monitoring"] = True
    else:
        # Configure monitoring interval
        interval = questionary.text(
            "Monitoring interval in seconds:",
            default="1.0",
            validate=lambda text: text.replace('.', '', 1).isdigit(),  # Validate float input
            style=custom_style
        ).ask()
        general_options["monitor_interval"] = float(interval)
    
    # Optional notes
    notes = questionary.text(
        "Add notes to the benchmark (optional):",
        style=custom_style
    ).ask()
    
    if notes:
        general_options["notes"] = notes
    
    return general_options

def build_command(benchmark, benchmark_options, general_options):
    """Build the command to run with all options"""
    # Start with the basic command
    cmd = [sys.executable, "wtf.py", "--benchmark", benchmark]
    
    # Add benchmark-specific options
    for option, value in benchmark_options.items():
        # Convert underscores to hyphens for command-line arguments
        cli_option = option.replace('_', '-')
        
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{cli_option}")
            else:
                cmd.append(f"--no-{cli_option}")
        elif isinstance(value, list):
            cmd.append(f"--{cli_option}")
            for item in value:
                cmd.append(str(item))
        else:
            cmd.append(f"--{cli_option}")
            cmd.append(str(value))
    
    # Add general options
    for option, value in general_options.items():
        # DO NOT convert underscores to hyphens for general options
        cli_option = option  # Use the option name directly
        
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{cli_option}")
            else:
                cmd.append(f"--no-{cli_option}")
        else:
            cmd.append(f"--{cli_option}")
            cmd.append(str(value))
    
    return cmd

def keyboard_monitor():
    """Monitor keyboard input for stop command ('s' key) - works on both Windows and Linux"""
    print("\n\033[1;33mPress 's' to stop the benchmark gracefully after current trial\033[0m")
    
    if IS_WINDOWS:
        # Windows-specific keyboard input handling
        while True:
            if msvcrt.kbhit():  # Check if a key was pressed
                key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                if key == 's':
                    if TERMINATE_REQUESTED:
                        print("\n\033[1;31mForcing benchmark termination...\033[0m")
                        sys.stdout.flush()
                        os._exit(0)  # Force immediate exit
                    else:
                        print("\n\033[1;33mStop requested. Completing current trial before stopping...\033[0m")
                        set_terminate()  # Call the function to set termination flag
            time.sleep(0.1)  # Prevent CPU hogging
    else:
        # Linux/Mac - need to set terminal to non-canonical mode
        import termios
        import tty
        
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while True:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()
                    if key == 's':
                        if TERMINATE_REQUESTED:
                            print("\n\033[1;31mForcing benchmark termination...\033[0m")
                            sys.stdout.flush()
                            os._exit(0)  # Force immediate exit
                        else:
                            print("\n\033[1;33mStop requested. Completing current trial before stopping...\033[0m")
                            set_terminate()  # Call the function to set termination flag
                time.sleep(0.1)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def check_termination():
    """Check if termination is requested and handle it gracefully"""
    last_trial_time = time.time()
    trials_since_termination = 0
    
    while True:
        if TERMINATE_REQUESTED:
            current_time = time.time()
            
            # If we see a new trial completing, record it
            if current_time - last_trial_time > 0.5:  # Assume a new trial if 0.5+ sec has passed
                last_trial_time = current_time
                trials_since_termination += 1
        time.sleep(0.5)

def run_interactive():
    """Run the benchmarking tool interactively"""
    # Skip GPU info display if fast-menu was used
    if not sys.argv[1:] or '--fast-menu' not in sys.argv[1:]:
        display_welcome()
    else:
        print("GPU Benchmark Tool - Fast Menu Mode")
    
    # Step 1: Select benchmark
    benchmark = select_benchmark()
    if not benchmark:
        return
    
    # Step 2: Configure benchmark-specific options
    benchmark_options = configure_benchmark(benchmark)
    
    # Step 3: Configure general options
    general_options = configure_general_options()
    
    # Step 4: Build the command to run
    cmd = build_command(benchmark, benchmark_options, general_options)
    
    # Display the command
    cmd_str = " ".join(cmd)
    rprint(f"\n[bold cyan]Command to execute:[/bold cyan] [yellow]{cmd_str}[/yellow]")
    
    # Ask for confirmation
    run = questionary.confirm(
        "Run the benchmark now?",
        default=True,
        style=custom_style
    ).ask()
    
    if run:
        rprint("\n[bold green]Starting benchmark...[/bold green]")
        # Run the benchmark using the constructed command
        subprocess.run(cmd)

def main():
    global TERMINATE_REQUESTED
    
    # Import torch only when ready to run the benchmark
    import torch
    
    # Check if no arguments were provided (interactive mode)
    if len(sys.argv) == 1:
        # No arguments, run interactive mode
        run_interactive()
        return
    
    # Command-line mode: Parse arguments
    parser = create_parser()
    
    # First pass to get the benchmark name
    temp_args, _ = parser.parse_known_args()
    
    # Import the module directly
    if temp_args.benchmark == "mamf":
        import benchmarks.mamf as benchmark_module
    elif temp_args.benchmark == "tensor":
        import benchmarks.tensor_ops as benchmark_module
    elif temp_args.benchmark == "datagen":
        import benchmarks.data_generation as benchmark_module
    elif temp_args.benchmark == "transfer":
        import benchmarks.transfer as benchmark_module
    elif temp_args.benchmark == "membw":
        import benchmarks.memory_bandwidth as benchmark_module
    elif temp_args.benchmark == "inference":
        import benchmarks.inference as benchmark_module
    elif temp_args.benchmark == "compute":
        import benchmarks.computation as benchmark_module
    else:
        print(f"Error: Benchmark '{temp_args.benchmark}' not available.")
        return
    
    # Add benchmark-specific arguments
    benchmark_module.add_benchmark_args(parser)
    
    # Final argument parsing with all options
    args = parser.parse_args()
    
    # Set default output file if not provided
    if args.output_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = str(RESULTS_DIR / "logs" / f"{args.benchmark}_{timestamp}.txt")
    else:
        # If output file is a relative path, put it in the results directory
        if not os.path.isabs(args.output_file):
            args.output_file = str(RESULTS_DIR / "logs" / args.output_file)
    
    # Set up output redirection to both console and file
    sys.stdout = Tee(args.output_file, args.verbose)
    
    # Get GPU information
    arch = get_accelerator_arch()
    device = arch.device()
    dtype = torch.bfloat16  # Default precision for benchmarks
    
    # Print benchmark header with system information
    print_benchmark_header(dtype, device, args.notes, arch)
    print(f"Results will be saved to: {args.output_file}")
    
    # Set up benchmark database path for all benchmarks
    benchmark_type = args.benchmark
    db_file = getattr(args, 'db_file', None)
    if db_file is None:
        db_file = f"{args.benchmark}.db"
    db_path = RESULTS_DIR / "db" / db_file
    print(f"Benchmark database file: {db_path}")
    
    # Initialize database for the benchmark type
    setup_database(db_path)
    
    # Update args with absolute database path
    args.db_file = str(db_path)
    
    # Set summary path for all benchmarks
    summary_path = RESULTS_DIR / "benchmark_summary.json"
    args.summary_file = str(summary_path)
    
    # Start GPU monitoring if enabled
    monitor = None
    if not args.skip_monitoring and arch.name() == "cuda":
        monitor_db_path = RESULTS_DIR / "db" / args.monitor_db
        print(f"GPU monitoring data will be saved to: {monitor_db_path}")
        
        try:
            monitor = GPUMonitor(monitor_db_path, interval=args.monitor_interval)
            
            args_dict = {k: v for k, v in vars(args).items()}
            monitor_id = monitor.start_monitoring(
                f"{args.benchmark}_benchmark", 
                args.benchmark,  
                parameters=json.dumps(args_dict)
            )
            print(f"GPU monitoring started (ID: {monitor_id})")
        except Exception as e:
            print(f"Warning: Failed to start GPU monitoring: {e}")
            monitor = None
    
    # Record start time for benchmark duration calculation
    start_time = time.time()
    
    # Start keyboard monitor thread to allow graceful stopping
    monitor_thread = threading.Thread(target=keyboard_monitor, daemon=True)
    monitor_thread.start()
    
    # Set up signal handler for ctrl+c
    def signal_handler(signum, frame):
        global TERMINATE_REQUESTED
        
        if TERMINATE_REQUESTED:
            print("\n\033[1;31mForcing immediate termination...\033[0m")
            # Force cleanup
            if monitor:
                try:
                    monitor.stop_monitoring()
                except:
                    pass
            sys.exit(1)
        else:
            TERMINATE_REQUESTED = True
            print("\n\033[1;33mStop requested. Completing current trial before stopping...\033[0m")
            print("\nTermination flag set - will stop after current trial")
            print("Press 's' again to force immediate termination")
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize the termination check thread
    termination_check = threading.Thread(target=check_termination, daemon=True)
    termination_check.start()
    
    # Run the selected benchmark
    if args.benchmark in AVAILABLE_BENCHMARKS:
        try:
            # Run benchmark
            result = benchmark_module.run_benchmark(args)
            
            # Process benchmark results if they were returned
            if isinstance(result, tuple) and len(result) == 2:
                summary_path = RESULTS_DIR / "benchmark_summary.json"
                
                # Skip summary update if already done in benchmark module
                if not hasattr(args, 'summary_updated') or not args.summary_updated:
                    # Load existing summary if available
                    summary = {}
                    if os.path.exists(summary_path):
                        try:
                            with open(summary_path, 'r') as f:
                                summary = json.load(f)
                        except json.JSONDecodeError:
                            summary = {}
                    
                    # Extract benchmark results
                    perf_data, config = result
                    
                    # Create result data for summary
                    device_info = arch.device_info()
                    
                    result_data = {
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "performance": perf_data.get('throughput_samples_per_sec', 0) or
                                       perf_data.get('bandwidth_gbps', 0) or
                                       perf_data.get('gflops', 0),
                        "config": str(config),
                        "device": str(device_info)
                    }
                    
                    # Update summary with the new result
                    from utils.db_utils import update_benchmark_summary
                    update_benchmark_summary(args.benchmark, result_data, summary_path)
                    
        except Exception as e:
            print(f"Error running benchmark: {e}")
        finally:
            # Always ensure monitoring is stopped properly
            if monitor:
                try:
                    print("Stopping GPU monitoring...")
                    monitor.stop_monitoring()
                except Exception as e:
                    print(f"Error stopping GPU monitoring: {e}")
            
            # Calculate and print total benchmark time
            time_delta = time.time() - start_time
            time_str = str(datetime.timedelta(seconds=time_delta)).split(".")[0]
            print(f"Total benchmark time: {time_str}")
    else:
        print(f"Error: Benchmark '{args.benchmark}' not available.")

if __name__ == "__main__":
    main()