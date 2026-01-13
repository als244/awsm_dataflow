import torch
import os
import subprocess

def get_available_gpu_memory(device_id=0):
    """
    Get available GPU memory in bytes for the specified device.
    Supports both NVIDIA (CUDA) and AMD (ROCm) GPUs.
    
    Args:
        device_id: GPU device index (default: 0)
    
    Returns:
        Available GPU memory in bytes
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
        
        # Get memory info using PyTorch (works for both CUDA and ROCm)
        free_memory, total_memory = torch.cuda.mem_get_info(device_id)
        return free_memory
    
    # Fallback: try to detect GPU type and use vendor-specific tools
    # Check for NVIDIA GPU using nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", f"--id={device_id}"],
            capture_output=True,
            text=True,
            check=True
        )
        # nvidia-smi returns memory in MiB
        free_memory_mib = int(result.stdout.strip())
        return free_memory_mib * 1024 * 1024  # Convert MiB to bytes
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Check for AMD GPU using rocm-smi
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            check=True
        )
        import json
        data = json.loads(result.stdout)
        # Parse rocm-smi JSON output for the specified device
        for card_key, card_info in data.items():
            if f"card{device_id}" in card_key.lower() or card_key == str(device_id):
                if "VRAM Total Used Memory (B)" in card_info and "VRAM Total Memory (B)" in card_info:
                    total = int(card_info["VRAM Total Memory (B)"])
                    used = int(card_info["VRAM Total Used Memory (B)"])
                    return total - used
        # Alternative parsing for different rocm-smi versions
        if "card0" in data or "GPU" in str(data):
            # Try to find free memory in various formats
            for key, value in data.items():
                if "free" in key.lower() and "vram" in key.lower():
                    return int(value)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    
    # Alternative AMD approach using /sys filesystem (Linux)
    try:
        amd_mem_path = f"/sys/class/drm/card{device_id}/device/mem_info_vram_total"
        amd_mem_used_path = f"/sys/class/drm/card{device_id}/device/mem_info_vram_used"
        if os.path.exists(amd_mem_path) and os.path.exists(amd_mem_used_path):
            with open(amd_mem_path, 'r') as f:
                total = int(f.read().strip())
            with open(amd_mem_used_path, 'r') as f:
                used = int(f.read().strip())
            return total - used
    except (IOError, ValueError):
        pass
    
    raise RuntimeError(
        f"Could not determine available GPU memory for device {device_id}. "
        "Ensure either CUDA/ROCm is properly installed with PyTorch support, "
        "or nvidia-smi/rocm-smi is available."
    )


def get_available_host_memory():
    """
    Get available host (CPU/system) memory in bytes.
    
    Accounts for:
    - SLURM memory allocations (via cgroups)
    - Container memory limits (Docker, Singularity, etc.)
    - System-wide available memory
    
    Returns:
        Available host memory in bytes
    """
    # First, check for cgroup memory limits (SLURM, containers, etc.)
    cgroup_limit = _get_cgroup_memory_limit()
    cgroup_usage = _get_cgroup_memory_usage()
    
    # Get system-wide available memory
    system_available = _get_system_available_memory()
    
    # If we're in a cgroup with a memory limit, calculate available within that limit
    if cgroup_limit is not None and cgroup_usage is not None:
        cgroup_available = cgroup_limit - cgroup_usage
        # Return the minimum of cgroup available and system available
        # (system might have less free memory than our cgroup limit allows)
        return min(cgroup_available, system_available)
    
    return system_available


def _get_cgroup_memory_limit():
    """
    Get the cgroup memory limit if one exists.
    Checks both cgroups v1 and v2.
    
    Returns:
        Memory limit in bytes, or None if no limit is set
    """
    # Try cgroups v2 first (newer systems, including newer SLURM versions)
    cgroup_v2_limit = "/sys/fs/cgroup/memory.max"
    if os.path.exists(cgroup_v2_limit):
        try:
            with open(cgroup_v2_limit, "r") as f:
                value = f.read().strip()
                if value == "max":
                    return None  # No limit set
                return int(value)
        except (IOError, ValueError):
            pass
    
    # Try cgroups v1 (older systems, many SLURM installations)
    cgroup_v1_limit = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    if os.path.exists(cgroup_v1_limit):
        try:
            with open(cgroup_v1_limit, "r") as f:
                value = int(f.read().strip())
                # Check if this is effectively "unlimited" 
                # (usually a very large number close to max int64)
                if value >= 9223372036854771712:  # Common "unlimited" value
                    return None
                return value
        except (IOError, ValueError):
            pass
    
    # Try to find the cgroup path from /proc/self/cgroup and check there
    cgroup_path = _find_memory_cgroup_path()
    if cgroup_path:
        # cgroups v2
        v2_path = os.path.join(cgroup_path, "memory.max")
        if os.path.exists(v2_path):
            try:
                with open(v2_path, "r") as f:
                    value = f.read().strip()
                    if value != "max":
                        return int(value)
            except (IOError, ValueError):
                pass
        
        # cgroups v1
        v1_path = os.path.join(cgroup_path, "memory.limit_in_bytes")
        if os.path.exists(v1_path):
            try:
                with open(v1_path, "r") as f:
                    value = int(f.read().strip())
                    if value < 9223372036854771712:
                        return value
            except (IOError, ValueError):
                pass
    
    return None


def _get_cgroup_memory_usage():
    """
    Get current cgroup memory usage.
    Checks both cgroups v1 and v2.
    
    Returns:
        Current memory usage in bytes, or None if not in a cgroup
    """
    # Try cgroups v2 first
    cgroup_v2_current = "/sys/fs/cgroup/memory.current"
    if os.path.exists(cgroup_v2_current):
        try:
            with open(cgroup_v2_current, "r") as f:
                return int(f.read().strip())
        except (IOError, ValueError):
            pass
    
    # Try cgroups v1
    cgroup_v1_usage = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
    if os.path.exists(cgroup_v1_usage):
        try:
            with open(cgroup_v1_usage, "r") as f:
                return int(f.read().strip())
        except (IOError, ValueError):
            pass
    
    # Try to find the cgroup path from /proc/self/cgroup
    cgroup_path = _find_memory_cgroup_path()
    if cgroup_path:
        # cgroups v2
        v2_path = os.path.join(cgroup_path, "memory.current")
        if os.path.exists(v2_path):
            try:
                with open(v2_path, "r") as f:
                    return int(f.read().strip())
            except (IOError, ValueError):
                pass
        
        # cgroups v1
        v1_path = os.path.join(cgroup_path, "memory.usage_in_bytes")
        if os.path.exists(v1_path):
            try:
                with open(v1_path, "r") as f:
                    return int(f.read().strip())
            except (IOError, ValueError):
                pass
    
    return None


def _find_memory_cgroup_path():
    """
    Find the memory cgroup path for the current process.
    
    Returns:
        Path to the memory cgroup, or None if not found
    """
    if not os.path.exists("/proc/self/cgroup"):
        return None
    
    try:
        with open("/proc/self/cgroup", "r") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) >= 3:
                    hierarchy_id, controllers, path = parts[0], parts[1], parts[2]
                    
                    # cgroups v2 (unified hierarchy)
                    if hierarchy_id == "0" and controllers == "":
                        cgroup_root = "/sys/fs/cgroup"
                        full_path = os.path.join(cgroup_root, path.lstrip("/"))
                        if os.path.exists(full_path):
                            return full_path
                    
                    # cgroups v1 (memory controller)
                    if "memory" in controllers.split(","):
                        cgroup_root = "/sys/fs/cgroup/memory"
                        full_path = os.path.join(cgroup_root, path.lstrip("/"))
                        if os.path.exists(full_path):
                            return full_path
    except IOError:
        pass
    
    return None


def _get_system_available_memory():
    """
    Get system-wide available memory.
    This is your original implementation.
    
    Returns:
        Available system memory in bytes
    """
    # Try using psutil first (most reliable cross-platform)
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.available
    except ImportError:
        pass
    
    # Linux: Read from /proc/meminfo
    if os.path.exists("/proc/meminfo"):
        try:
            meminfo = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        value = int(parts[1])
                        # Values in /proc/meminfo are in kB
                        meminfo[key] = value * 1024
            
            # MemAvailable is the best metric (available since Linux 3.14)
            if "MemAvailable" in meminfo:
                return meminfo["MemAvailable"]
            
            # Fallback: estimate available memory
            if all(k in meminfo for k in ["MemFree", "Buffers", "Cached"]):
                return meminfo["MemFree"] + meminfo["Buffers"] + meminfo["Cached"]
            
            if "MemFree" in meminfo:
                return meminfo["MemFree"]
        except (IOError, ValueError, KeyError):
            pass
    
    # macOS: Use vm_stat
    try:
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            check=True
        )
        page_size = 4096
        try:
            pagesize_result = subprocess.run(
                ["pagesize"],
                capture_output=True,
                text=True,
                check=True
            )
            page_size = int(pagesize_result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            pass
        
        stats = {}
        for line in result.stdout.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                value = value.strip().rstrip(".")
                try:
                    stats[key.strip()] = int(value)
                except ValueError:
                    continue
        
        free_pages = stats.get("Pages free", 0)
        inactive_pages = stats.get("Pages inactive", 0)
        speculative_pages = stats.get("Pages speculative", 0)
        return (free_pages + inactive_pages + speculative_pages) * page_size
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Windows: Use wmic
    try:
        result = subprocess.run(
            ["wmic", "OS", "get", "FreePhysicalMemory", "/value"],
            capture_output=True,
            text=True,
            check=True
        )
        for line in result.stdout.split("\n"):
            if "FreePhysicalMemory" in line:
                value = int(line.split("=")[1].strip())
                return value * 1024
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, IndexError):
        pass
    
    raise RuntimeError(
        "Could not determine available host memory. "
        "Consider installing psutil: pip install psutil"
    )


def get_memory_info():
    """
    Get detailed memory information including limits and usage.
    Useful for debugging and logging.
    
    Returns:
        Dictionary with memory information
    """
    info = {
        "system_available": _get_system_available_memory(),
        "cgroup_limit": _get_cgroup_memory_limit(),
        "cgroup_usage": _get_cgroup_memory_usage(),
        "effective_available": get_available_host_memory(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
    }
    return info