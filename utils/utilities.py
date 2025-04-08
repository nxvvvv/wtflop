import datetime
import platform
import shlex
import sys
import termios
import time
import tty
import torch

def print_benchmark_header(dtype, device, notes="None", arch=None):
    """Print a formatted header for the benchmark"""
    if arch is None:
        from utils.arch import get_accelerator_arch
        arch = get_accelerator_arch()
        
    device_info = arch.device_info()
    compute_info = arch.compute_info()

    print(
        f"""
Benchmark started on {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}

** Command line:
{sys.executable} {" ".join(map(shlex.quote, sys.argv))}

** Dtype: {dtype}

** Platform/Device info:
{" ".join(platform.uname())}
{device_info}

** Critical software versions:
torch={torch.__version__}
{compute_info}

** Additional notes:
{notes}

{"-" * 80}

"""
    )

def getch():
    """Get a single character from stdin, Unix version"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def handle_sigkill(start_time):
    """Handler for SIGINT (Ctrl+C) to gracefully exit benchmarks"""
    time_delta = time.time() - start_time
    time_str = str(datetime.timedelta(seconds=time_delta)).split(".")[0]
    print(f"\nBenchmark interrupted after {time_str}")
    sys.exit(1)

def should_terminate():
    """Check if termination has been requested"""
    # Import the global flag from the main script
    from wtf import TERMINATE_REQUESTED
    return TERMINATE_REQUESTED