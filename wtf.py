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
import torch
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


# Set up directory structure
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = ROOT_DIR / "results"

# Create necessary directories if they don't exist
RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "logs").mkdir(exist_ok=True)  # For benchmark output logs
(RESULTS_DIR / "db").mkdir(exist_ok=True)    # For database files
(RESULTS_DIR / "reports").mkdir(exist_ok=True)  # For generated reports


# Import available benchmarks
try:
    from benchmarks import mamf, tensor_ops
    AVAILABLE_BENCHMARKS = {
        "mamf": mamf.run_benchmark,        # Matrix multiplication benchmark
        "tensor": tensor_ops.run_benchmark, # Tensor operations benchmark
    }
except ImportError as e:
    print(f"Warning: Could not import a benchmark module: {e}")
    AVAILABLE_BENCHMARKS = {}


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
        "--output_file", 
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
        "--monitor_interval",
        type=float,
        default=1.0,
        help="Interval in seconds between GPU monitoring samples"
    )
    parser.add_argument(
        "--monitor_db",
        type=str,
        default="gpu_monitoring.db",
        help="Database file to store GPU monitoring data"
    )
    parser.add_argument(
        "--skip_monitoring",
        action="store_true",
        help="Skip GPU monitoring during benchmarks"
    )
    
    return parser

def get_benchmark_specific_options(benchmark_name):
    """Get benchmark-specific options"""
    # Create a temporary parser for benchmark-specific options
    parser = argparse.ArgumentParser()
    
    # Add benchmark-specific arguments
    if benchmark_name == "mamf":
        mamf.add_benchmark_args(parser)
    elif benchmark_name == "tensor":
        tensor_ops.add_benchmark_args(parser)
        
    # Extract option details for interactive UI
    options = {}
    for action in parser._actions:
        if action.dest != 'help':
            options[action.dest] = {
                'help': action.help,
                'default': action.default,
                'type': action.type.__name__ if hasattr(action, 'type') and action.type else 'bool',
                'choices': action.choices,
                'nargs': action.nargs
            }
    
    return options

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
    """Let user select a benchmark to run"""
    if not AVAILABLE_BENCHMARKS:
        console.print("[bold red]No benchmarks available![/bold red]")
        return None
    
    # Display available benchmarks in a table
    table = Table(title="Available Benchmarks")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Description", style="green")
    
    descriptions = {
        "mamf": "Matrix Multiplication Benchmark (tests raw computational throughput)",
        "tensor": "Tensor Operations Benchmark (tests various tensor operations)"
    }
    
    for benchmark in AVAILABLE_BENCHMARKS.keys():
        table.add_row(benchmark, descriptions.get(benchmark, ""))
    
    console.print(table)
    
    # Let user select a benchmark
    selected = questionary.select(
        "Select a benchmark to run:",
        choices=list(AVAILABLE_BENCHMARKS.keys()),
        style=custom_style
    ).ask()
    
    return selected

def configure_benchmark(benchmark_name):
    """Configure benchmark options interactively"""
    # Get available options for this benchmark
    options = get_benchmark_specific_options(benchmark_name)
    
    # Display options in a table
    table = Table(title=f"{benchmark_name.upper()} Benchmark Options")
    table.add_column("Option", style="cyan")
    table.add_column("Default", style="yellow")
    table.add_column("Description", style="green")
    
    for name, details in options.items():
        default_str = str(details['default'])
        if isinstance(details['default'], list):
            default_str = str(details['default']).replace(',', ', ')
        table.add_row(name, default_str, details['help'])
    
    console.print(table)
    
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
        choices=list(options.keys()),
        style=custom_style
    ).ask()
    
    # Configure each selected option
    custom_options = {}
    
    for option in to_customize:
        details = options[option]
        
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
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{option}")
            else:
                cmd.append(f"--no-{option}")
        elif isinstance(value, list):
            cmd.append(f"--{option}")
            for item in value:
                cmd.append(str(item))
        else:
            cmd.append(f"--{option}")
            cmd.append(str(value))
    
    # Add general options
    for option, value in general_options.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{option}")
            else:
                cmd.append(f"--no-{option}")
        else:
            cmd.append(f"--{option}")
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
    display_welcome()
    
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
    
    # Check if no arguments were provided (interactive mode)
    if len(sys.argv) == 1:
        # No arguments, run interactive mode
        run_interactive()
        return
    
    # Command-line mode: Parse arguments
    parser = create_parser()
    
    # First pass to get the benchmark name
    temp_args, _ = parser.parse_known_args()
    
    # Add benchmark-specific arguments based on selected benchmark
    if temp_args.benchmark == "mamf":
        mamf.add_benchmark_args(parser)
    elif temp_args.benchmark == "tensor":
        tensor_ops.add_benchmark_args(parser)
    
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
    
    # Set up benchmark database path
    if args.benchmark == "mamf":
        db_file = getattr(args, 'db_file', 'mamf.db')
        db_path = RESULTS_DIR / "db" / db_file
        print(f"Benchmark database file: {db_path}")
    elif args.benchmark == "tensor":
        db_file = getattr(args, 'db_file', 'tensor_ops.db')
        db_path = RESULTS_DIR / "db" / db_file
        print(f"Benchmark database file: {db_path}")
    
    # Update args with absolute database path
    args.db_file = str(db_path)
    
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
        TERMINATE_REQUESTED = True
        if monitor:
            print("Stopping GPU monitoring...")
            monitor.stop_monitoring()
        handle_sigkill(start_time)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize the termination check thread
    termination_check = threading.Thread(target=check_termination, daemon=True)
    termination_check.start()
    
    # Run the selected benchmark
    if args.benchmark in AVAILABLE_BENCHMARKS:
        try:
            # Run benchmark WITHOUT passing termination_flag directly
            # Instead, use the global variable that the benchmark functions can check
            result = AVAILABLE_BENCHMARKS[args.benchmark](args)
            
            # If we got here and termination was requested, but benchmark completed anyway,
            # we can just exit now
            if TERMINATE_REQUESTED:
                print("\n\033[1;33mBenchmark completed after termination request.\033[0m")
                if monitor:
                    monitor.stop_monitoring()
                sys.exit(0)
            
            # Process benchmark results if they were returned
            if isinstance(result, tuple) and len(result) == 2:
                summary_path = RESULTS_DIR / "benchmark_summary.json"
                
                # Load existing summary if available
                summary = {}
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, 'r') as f:
                            summary = json.load(f)
                    except json.JSONDecodeError:
                        summary = {}
                
                # Add new benchmark result to summary
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if args.benchmark not in summary:
                    summary[args.benchmark] = []
                
                summary[args.benchmark].append({
                    "timestamp": timestamp,
                    "performance": result[0],
                    "config": result[1],
                    "device": str(arch.device_info())
                })
                
                # Save updated summary
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                print(f"Benchmark result added to summary: {summary_path}")
                
        except Exception as e:
            print(f"Error running benchmark: {e}")
    else:
        print(f"Error: Benchmark '{args.benchmark}' not available.")
    
    # Stop GPU monitoring if it was started
    if monitor:
        monitor.stop_monitoring()
        print("GPU monitoring stopped")
    
    # Calculate and print total benchmark time
    time_delta = time.time() - start_time
    time_str = str(datetime.timedelta(seconds=time_delta)).split(".")[0]
    print(f"Total benchmark time: {time_str}")

if __name__ == "__main__":
    main()