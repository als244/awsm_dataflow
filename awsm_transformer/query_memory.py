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
            pass # Fallback if torch fails
    
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
            with open(amd_mem_path, 'r') as f: total = int(f.read().strip())
            with open(amd_used_path, 'r') as f: used = int(f.read().strip())
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
        # If we know the Slurm limit, we must estimate usage to find 'available'.
        # Since standard tools report global usage, we rely on cgroup usage if possible.
        cgroup_usage = _get_cgroup_memory_usage()
        
        if cgroup_usage:
             # Calculate available based on Slurm limit and Cgroup usage
            available = slurm_limit - cgroup_usage
            # Handle edge case where usage > limit (swapping or soft limits)
            return max(0, available)
        else:
            # If we can't find cgroup usage, we return the Slurm limit 
            # minus a safety buffer (e.g. assume 10% overhead or use system load)
            # But safer to just return the Slurm limit if the job just started.
            # Ideally, if we are in Slurm, we are in a Cgroup.
            return slurm_limit

    # 2. Check Cgroup Limits directly (e.g. Docker/Kubernetes/Slurm without env vars)
    cgroup_limit = _get_cgroup_memory_limit()
    cgroup_usage = _get_cgroup_memory_usage()

    if cgroup_limit and cgroup_usage:
        cgroup_available = cgroup_limit - cgroup_usage
        return min(cgroup_available, system_available)

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
            # We take the first number as a rough estimate of allocated cores
            import re
            match = re.match(r'(\d+)', cpus_on_node)
            if match:
                num_cpus = int(match.group(1))
                return int(mem_per_cpu) * 1024 * 1024 * num_cpus
        except ValueError:
            pass
            
    return None


def _get_cgroup_memory_limit():
    """Get the memory limit from cgroups (v1 or v2)."""
    # 1. Try Cgroup V2
    if os.path.isfile("/sys/fs/cgroup/memory.max"):
        try:
            with open("/sys/fs/cgroup/memory.max", "r") as f:
                val = f.read().strip()
                if val != "max":
                    return int(val)
        except (IOError, ValueError): pass

    # 2. Try Cgroup V1
    if os.path.isfile("/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
                val = int(f.read().strip())
                # Filter out "unlimited" large numbers
                if val < 2**60: 
                    return val
        except (IOError, ValueError): pass
        
    # 3. Try finding process specific path
    path = _find_cgroup_path("memory")
    if path:
        # V2
        if os.path.isfile(os.path.join(path, "memory.max")):
            try:
                with open(os.path.join(path, "memory.max"), "r") as f:
                    val = f.read().strip()
                    if val != "max": return int(val)
            except: pass
        # V1
        if os.path.isfile(os.path.join(path, "memory.limit_in_bytes")):
            try:
                with open(os.path.join(path, "memory.limit_in_bytes"), "r") as f:
                    val = int(f.read().strip())
                    if val < 2**60: return val
            except: pass
            
    return None


def _get_cgroup_memory_usage():
    """
    Get current cgroup memory usage (Used - Cache).
    We subtract cache because Linux treats cache as 'used' memory, 
    but it is reclaimable if the app needs it.
    """
    usage = None
    cache = 0
    
    # 1. Try Cgroup V2
    if os.path.isfile("/sys/fs/cgroup/memory.current"):
        try:
            with open("/sys/fs/cgroup/memory.current", "r") as f:
                usage = int(f.read().strip())
            # Get cache info to subtract
            if os.path.isfile("/sys/fs/cgroup/memory.stat"):
                with open("/sys/fs/cgroup/memory.stat", "r") as f:
                    for line in f:
                        if line.startswith("file "): # V2 usually uses 'file' for page cache
                            cache = int(line.split()[1])
                            break
        except: pass

    # 2. Try Cgroup V1
    elif os.path.isfile("/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            with open("/sys/fs/cgroup/memory/memory.usage_in_bytes", "r") as f:
                usage = int(f.read().strip())
            if os.path.isfile("/sys/fs/cgroup/memory/memory.stat"):
                with open("/sys/fs/cgroup/memory/memory.stat", "r") as f:
                    for line in f:
                        if line.startswith("total_cache") or line.startswith("cache"):
                            cache = int(line.split()[1])
                            # Don't break immediately, look for total_cache preference
                            if line.startswith("total_cache"): break
        except: pass
        
    if usage is not None:
        return max(0, usage - cache)
        
    return None


def _find_cgroup_path(controller):
    """Finds the cgroup path for the current process via /proc/self/cgroup"""
    try:
        with open("/proc/self/cgroup", "r") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 3:
                    # parts[1] is controllers, parts[2] is path
                    if controller in parts[1] or (parts[1] == "" and parts[0] == "0"): # V1 or V2
                        # The path in /proc is relative to the mount point
                        rel_path = parts[2].lstrip("/")
                        
                        # Guess mount points
                        v2_path = os.path.join("/sys/fs/cgroup", rel_path)
                        v1_path = os.path.join(f"/sys/fs/cgroup/{controller}", rel_path)
                        
                        if os.path.exists(v2_path): return v2_path
                        if os.path.exists(v1_path): return v1_path
    except:
        pass
    return None


def _get_system_available_memory():
    """Fallback: Standard system memory check"""
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
            except: pass
    return 0

# Test print
if __name__ == "__main__":
    host_mem = get_available_host_memory()
    gpu_mem = get_available_gpu_memory()
    
    print(f"Host Memory Available: {host_mem / 1024**3:.2f} GB")
    print(f"GPU Memory Available:  {gpu_mem / 1024**3:.2f} GB")
    
    # Debug info
    if os.environ.get("SLURM_JOB_ID"):
        print(f"Running inside Slurm Job: {os.environ.get('SLURM_JOB_ID')}")
        print(f"Slurm Node Limit: {os.environ.get('SLURM_MEM_PER_NODE')} MB")