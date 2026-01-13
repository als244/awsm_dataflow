import ctypes
import numpy as np
import os
import glob
import sys

class TransmissionScheduler:
    def __init__(self):
        # 1. Find the compiled shared library
        current_dir = os.path.dirname(os.path.abspath(__file__))
        patterns = [
            os.path.join(current_dir, "_capi*.so"),      # Linux/Unix
            os.path.join(current_dir, "_capi*.dylib"),   # MacOS
            os.path.join(current_dir, "_capi*.pyd"),     # Windows
        ]
        
        lib_path = None
        for p in patterns:
            matches = glob.glob(p)
            if matches:
                lib_path = matches[0]
                break
        
        if not lib_path:
            raise FileNotFoundError(f"Could not find compiled C extension '_capi' in {current_dir}.")

        # 2. Load Library
        try:
            self.lib = ctypes.CDLL(lib_path)
        except OSError as e:
            raise OSError(f"Failed to load C library at {lib_path}: {e}")

        # 3. Define Signatures
        self.lib.solve_scheduler.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
            ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C')
        ]
        self.lib.solve_scheduler.restype = ctypes.c_double

    def solve(self, compute, durations, sizes, N, deadline, alpha=2.0):
        """
        Returns:
            best_val (float): Total Utility (Score)
            best_choices (int32 array): The option indices
            best_idle (float): Total transmitter idle time (ms)
        """
        # Calculate 'Efficiency Score' (Size - Time Penalty)
        effective_sizes = sizes - (durations * alpha)

        # 1. Setup Inputs
        compute = np.ascontiguousarray(compute, dtype=np.float64)
        durations = np.ascontiguousarray(durations, dtype=np.float64)
        sizes = np.ascontiguousarray(effective_sizes, dtype=np.float64)
        
        current_compute = compute.copy()
        
        # Best result containers
        best_val = 0.0
        best_choices = np.zeros(len(compute), dtype=np.int32)
        best_idle = 0.0
        
        # Iteration Loop (Fixed Point Iteration for Buffer Constraints)
        for pass_idx in range(3):
            val, choices = self._run_c_solver(current_compute, durations, sizes, N, deadline)
            
            if val == 0.0:
                # Return failure consistent signature
                return 0.0, choices, 0.0
            
            # --- RECONSTRUCTION & IDLE CALC ---
            finish_times = np.zeros(len(compute))
            raw_arrivals = np.cumsum(current_compute)
            
            current_time = 0.0
            pass_total_idle = 0.0  # Track idle time for this pass
            
            for i in range(len(compute)):
                arrival_time = raw_arrivals[i]
                
                # Earliest the task can start (Physically)
                start_time = max(current_time, arrival_time)
                
                # Check Buffer Constraint (Blocking)
                if i >= N:
                    buffer_open_time = finish_times[i-N]
                    if buffer_open_time > start_time:
                         start_time = buffer_open_time
                
                # Capture Idle Time
                # If start_time > current_time, the transmitter was empty/waiting
                if start_time > current_time:
                    pass_total_idle += (start_time - current_time)
                
                # Transmission
                dur = durations[i, choices[i]]
                finish_times[i] = start_time + dur
                current_time = finish_times[i]
            
            # Update Bests
            best_val = val
            best_choices = choices
            best_idle = pass_total_idle
            
            # --- NEXT PASS SETUP ---
            # Calculate forced delays for next iteration
            new_arrivals = np.zeros(len(compute))
            base_arrivals = np.cumsum(compute) # Original physical arrivals
            
            for i in range(len(compute)):
                eff_arrival = base_arrivals[i]
                if i >= N:
                    if finish_times[i-N] > eff_arrival:
                        eff_arrival = finish_times[i-N]
                new_arrivals[i] = eff_arrival
            
            new_compute = np.zeros(len(compute))
            new_compute[0] = new_arrivals[0]
            new_compute[1:] = np.diff(new_arrivals)
            
            if np.allclose(current_compute, new_compute, atol=1e-3):
                break
                
            current_compute = new_compute

        return best_val, best_choices, best_idle

    def _run_c_solver(self, compute, durations, sizes, N, deadline):
        T, k = durations.shape
        flat_durs = durations.flatten()
        flat_sizes = sizes.flatten()
        choices = np.zeros(T, dtype=np.int32)
        val = self.lib.solve_scheduler(
            T, N, k, compute, flat_durs, flat_sizes, deadline, choices
        )
        return val, choices