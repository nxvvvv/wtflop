import platform
import subprocess
import time
import torch

has_hpu = False
try:
    import habana_frameworks.torch as ht
    if torch.hpu.is_available():
        has_hpu = True
except ModuleNotFoundError:
    pass


def clear_l2_cache():
    """Allocate tensor larger than L2 cache to clear it"""
    # Modern GPUs typically have 512KB to 6MB L2 cache
    cache_size_mb = 100  # 100MB to be safe
    n_elements = int(cache_size_mb * 1024 * 1024 / 4)  # Divide by 4 bytes (float32)

    # Create and fill tensor
    dummy = torch.ones(n_elements, dtype=torch.float32, device="cuda")
    dummy += 1  # Force memory access

    # Synchronize to ensure operation is complete
    torch.cuda.synchronize()


class Arch:
    """Base class for all accelerator architectures"""
    def __init__(self):
        self.arch = "unknown"

    def __repr__(self):
        return self.arch

    def clear_cache(self):
        pass


class CUDAArch(Arch):
    """shared with CUDA and ROCm: NVIDIA + AMD"""

    def __init__(self):
        if torch.version.hip is not None:
            self.arch = "rocm"
        else:
            self.arch = "cuda"

    def device(self):
        return torch.device("cuda:0")

    def name(self):
        return self.arch

    def device_info(self):
        return torch.cuda.get_device_properties(self.device())

    def compute_info(self):
        if self.arch == "rocm":
            return f"hip={torch.version.hip}, cuda={torch.version.cuda}"
        else:
            return f"cuda={torch.version.cuda}"

    def event(self, enable_timing=True):
        return torch.cuda.Event(enable_timing)

    def synchronize(self):
        torch.cuda.synchronize()

    def clear_cache(self):
        clear_l2_cache()


class HPUArch(Arch):
    """Intel Gaudi*"""

    def __init__(self):
        self.arch = "hpu"

    def device(self):
        return torch.device("hpu")

    def name(self):
        return self.arch

    def device_info(self):
        return torch.hpu.get_device_properties(self.device())

    def compute_info(self):
        return f"hpu={torch.version.hpu}"

    def event(self, enable_timing=True):
        return ht.hpu.Event(enable_timing)

    def synchronize(self):
        ht.hpu.synchronize()


class MPSEvent:
    def __init__(self):
        self.time = None

    def record(self):
        self.time = time.perf_counter()

    def elapsed_time(self, end):
        if self.time is None or end.time is None:
            return None
        return (end.time - self.time) * 1000  # Convert to milliseconds

    @staticmethod
    def synchronize():
        torch.mps.synchronize()


class MetalArch(Arch):
    """Apple Silicon"""

    def __init__(self):
        self.arch = "metal"

    def device(self):
        return torch.device("mps")

    def name(self):
        return self.arch

    def device_info(self):
        if platform.system() != "Darwin":
            return "Not a Mac"

        # Run system_profiler command to get hardware info
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"], capture_output=True, text=True
        )

        # Parse the output
        for line in result.stdout.split("\n"):
            if "Chip" in line:
                chip_info = line.split(":")[1].strip()
                return chip_info

    def compute_info(self):
        return "metal"

    def event(self, enable_timing=True):
        return MPSEvent()

    def synchronize(self):
        torch.mps.synchronize()


# Singleton-like approach for architecture
_arch_instance = None

def get_accelerator_arch():
    """
    Returns a singleton architecture instance
    (CUDAArch, HPUArch, or MetalArch object)
    """
    global _arch_instance
    if _arch_instance is not None:
        return _arch_instance
    
    # cuda / rocm
    if torch.cuda.is_available():
        _arch_instance = CUDAArch()
        return _arch_instance

    # hpu
    if has_hpu:
        _arch_instance = HPUArch()
        return _arch_instance

    # Apple Silicon
    if torch.backends.mps.is_available():
        _arch_instance = MetalArch()
        return _arch_instance

    raise ValueError("Currently only cuda, rocm, hpu and metal are supported")