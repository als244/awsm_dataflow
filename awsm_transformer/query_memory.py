import torch
import os
import subprocess
import sys


def get_available_gpu_memory(device_id=0):
    """
    Get available GPU memory in bytes for the specified device.
    Supports both NVIDIA (CUDA) and AMD (ROCm) GPUs.
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
        # Get memory info using PyTorch
        try:
            free_memory, total_memory = torch.cuda.mem_get_info(device_id)
            return free_memory
        except Exception:
            pass  # Fallback if torch fails

    # Fallback: NVIDIA
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", f"--id={device_id}"],
            capture_output=True, text=True, check=True
        )
        return int(result.stdout.strip()) * 1024 * 1024
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass

    # Fallback: AMD (ROCm)
    try:
        # Check sysfs first (lighter weight)
        amd_mem_path = f"/sys/class/drm/card{device_id}/device/mem_info_vram_total"
        amd_used_path = f"/sys/class/drm/card{device_id}/device/mem_info_vram_used"
        if os.path.exists(amd_mem_path) and os.path.exists(amd_used_path):
            with open(amd_mem_path, 'r') as f:
                total = int(f.read().strip())
            with open(amd_used_path, 'r') as f:
                used = int(f.read().strip())
            return total - used

        # Try rocm-smi
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, check=True
        )
        import json
        data = json.loads(result.stdout)
        for key, val in data.items():
            if f"card{device_id}" in key.lower() or key == str(device_id):
                if "VRAM Total Memory (B)" in val and "VRAM Total Used Memory (B)" in val:
                    return int(val["VRAM Total Memory (B)"]) - int(val["VRAM Total Used Memory (B)"])
    except Exception:
        pass

    print(f"Warning: Could not determine GPU memory for device {device_id}. Returning 0.")
    return 0


# -----------------------------------------------------------------------------
# HOST MEMORY FUNCTIONS
# -----------------------------------------------------------------------------

def get_available_host_memory():
    """
    Get available host (CPU/system) memory in bytes.

    Priority of checks:
    1. Slurm Environment Variables (Direct allocation info)
    2. Cgroup Limits (Kernel enforcement)
    3. System Available (Fallback)
    """
    system_available = _get_system_available_memory()

    # 1. Check Slurm Allocation
    slurm_limit = _get_slurm_memory_limit()
    if slurm_limit:
        cgroup_usage = _get_cgroup_memory_usage()

        if cgroup_usage is not None:
            available = slurm_limit - cgroup_usage
            return max(0, available)
        else:
            # Cgroup usage unavailable — estimate from process tree or /proc
            process_usage = _get_process_tree_rss()
            if process_usage is not None:
                return max(0, slurm_limit - process_usage)
            return slurm_limit

    # 2. Check Cgroup Limits directly (e.g. Docker/Kubernetes/Slurm without env vars)
    cgroup_limit = _get_cgroup_memory_limit()
    cgroup_usage = _get_cgroup_memory_usage()

    if cgroup_limit and cgroup_usage is not None:
        cgroup_available = cgroup_limit - cgroup_usage
        return min(max(0, cgroup_available), system_available)

    # 3. Fallback to whole system memory
    return system_available


def _get_slurm_memory_limit():
    """
    Reads SLURM environment variables to find the memory limit.
    Returns bytes or None.
    """
    # Case A: Explicit memory per node (most common)
    # SLURM_MEM_PER_NODE is usually in MB
    mem_per_node = os.environ.get("SLURM_MEM_PER_NODE")
    if mem_per_node:
        try:
            return int(mem_per_node) * 1024 * 1024
        except ValueError:
            pass

    # Case B: Memory per CPU
    # SLURM_MEM_PER_CPU is in MB
    mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    cpus_on_node = os.environ.get("SLURM_JOB_CPUS_PER_NODE")

    if mem_per_cpu and cpus_on_node:
        try:
            # SLURM_JOB_CPUS_PER_NODE format can be "4" or "4(x2)" etc.
            import re
            match = re.match(r'(\d+)', cpus_on_node)
            if match:
                num_cpus = int(match.group(1))
                return int(mem_per_cpu) * 1024 * 1024 * num_cpus
        except ValueError:
            pass

    return None


# -----------------------------------------------------------------------------
# CGROUP HELPERS (shared path resolution)
# -----------------------------------------------------------------------------

def _resolve_cgroup_memory_paths():
    """
    Find the correct cgroup memory directory.
    Returns (version, path) where version is "v1" or "v2", or (None, None).
    
    This ensures both limit and usage functions read from the same cgroup,
    avoiding mismatches (e.g. in Slurm where jobs live in nested cgroups).
    """
    # V2 root
    if os.path.isfile("/sys/fs/cgroup/memory.current"):
        return ("v2", "/sys/fs/cgroup")

    # V1 root
    if os.path.isfile("/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        return ("v1", "/sys/fs/cgroup/memory")

    # Process-specific path (e.g. Slurm job cgroups like
    # /sys/fs/cgroup/memory/slurm/uid_1000/job_12345/)
    path = _find_cgroup_path("memory")
    if path:
        if os.path.isfile(os.path.join(path, "memory.current")):
            return ("v2", path)
        if os.path.isfile(os.path.join(path, "memory.usage_in_bytes")):
            return ("v1", path)

    return (None, None)


def _get_cgroup_memory_limit():
    """Get the memory limit from cgroups (v1 or v2)."""
    version, path = _resolve_cgroup_memory_paths()
    if version is None:
        return None

    try:
        if version == "v2":
            limit_file = os.path.join(path, "memory.max")
            if os.path.isfile(limit_file):
                with open(limit_file, "r") as f:
                    val = f.read().strip()
                    if val != "max":
                        return int(val)
        elif version == "v1":
            limit_file = os.path.join(path, "memory.limit_in_bytes")
            if os.path.isfile(limit_file):
                with open(limit_file, "r") as f:
                    val = int(f.read().strip())
                    # Filter out "unlimited" large numbers
                    if val < 2**60:
                        return val
    except (IOError, ValueError):
        pass

    return None


def _get_cgroup_memory_usage():
    """
    Get current cgroup memory usage (Used - Cache).
    We subtract cache because Linux treats cache as 'used' memory,
    but it is reclaimable if the app needs it.
    
    Returns bytes or None.
    """
    version, path = _resolve_cgroup_memory_paths()
    if version is None:
        return None

    usage = None
    cache = 0

    try:
        if version == "v2":
            current_file = os.path.join(path, "memory.current")
            stat_file = os.path.join(path, "memory.stat")
            if os.path.isfile(current_file):
                with open(current_file, "r") as f:
                    usage = int(f.read().strip())
                if os.path.isfile(stat_file):
                    with open(stat_file, "r") as f:
                        for line in f:
                            if line.startswith("file "):
                                cache = int(line.split()[1])
                                break

        elif version == "v1":
            usage_file = os.path.join(path, "memory.usage_in_bytes")
            stat_file = os.path.join(path, "memory.stat")
            if os.path.isfile(usage_file):
                with open(usage_file, "r") as f:
                    usage = int(f.read().strip())
                if os.path.isfile(stat_file):
                    with open(stat_file, "r") as f:
                        for line in f:
                            if line.startswith("total_cache") or line.startswith("cache"):
                                cache = int(line.split()[1])
                                if line.startswith("total_cache"):
                                    break
    except (IOError, ValueError):
        pass

    if usage is not None:
        return max(0, usage - cache)

    return None


def _find_cgroup_path(controller):
    """Finds the cgroup path for the current process via /proc/self/cgroup."""
    try:
        with open("/proc/self/cgroup", "r") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 3:
                    # parts[1] is controllers, parts[2] is path
                    if controller in parts[1] or (parts[1] == "" and parts[0] == "0"):  # V1 or V2
                        rel_path = parts[2].lstrip("/")

                        # Guess mount points
                        v2_path = os.path.join("/sys/fs/cgroup", rel_path)
                        v1_path = os.path.join(f"/sys/fs/cgroup/{controller}", rel_path)

                        if os.path.exists(v2_path):
                            return v2_path
                        if os.path.exists(v1_path):
                            return v1_path
    except (IOError, OSError):
        pass
    return None


# -----------------------------------------------------------------------------
# FALLBACK HELPERS
# -----------------------------------------------------------------------------

def _get_process_tree_rss():
    """
    Estimate memory usage from the current process tree's RSS.
    Used as a fallback when cgroup usage is unavailable.
    Note: This only captures the current process and its children,
    not other processes in the same Slurm job.
    """
    # Try psutil first (captures full process tree)
    try:
        import psutil
        process = psutil.Process()
        usage = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                usage += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return usage
    except ImportError:
        pass

    # Fallback: /proc/self/status (current process only)
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024  # kB to bytes
    except (IOError, ValueError):
        pass

    return None


def _get_system_available_memory():
    """Fallback: Standard system memory check."""
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        # Simple linux fallback
        if os.path.isfile("/proc/meminfo"):
            try:
                with open("/proc/meminfo", "r") as f:
                    meminfo = {}
                    for line in f:
                        parts = line.split()
                        meminfo[parts[0].rstrip(":")] = int(parts[1]) * 1024
                    if "MemAvailable" in meminfo:
                        return meminfo["MemAvailable"]
                    return meminfo.get("MemFree", 0) + meminfo.get("Cached", 0)
            except (IOError, ValueError):
                pass
    return 0


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    host_mem = get_available_host_memory()
    gpu_mem = get_available_gpu_memory()

    print(f"Host Memory Available: {host_mem / 1024**3:.2f} GB")
    print(f"GPU Memory Available:  {gpu_mem / 1024**3:.2f} GB")

    # Debug info
    print(f"\n--- Debug ---")
    cg_version, cg_path = _resolve_cgroup_memory_paths()
    print(f"Cgroup version: {cg_version}, path: {cg_path}")
    print(f"Cgroup limit:   {_get_cgroup_memory_limit()}")
    print(f"Cgroup usage:   {_get_cgroup_memory_usage()}")
    print(f"Slurm limit:    {_get_slurm_memory_limit()}")
    print(f"System avail:   {_get_system_available_memory() / 1024**3:.2f} GB")

    if os.environ.get("SLURM_JOB_ID"):
        print(f"\nRunning inside Slurm Job: {os.environ.get('SLURM_JOB_ID')}")
        print(f"Slurm Node Limit: {os.environ.get('SLURM_MEM_PER_NODE')} MB")