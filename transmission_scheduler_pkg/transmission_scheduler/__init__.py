import ctypes
import numpy as np
import os
import glob
import sys

class TransmissionScheduler:
    def __init__(self):
        # 1. Find the compiled shared library
        # It will be named something like '_capi.cpython-39-x86_64-linux-gnu.so'
        # and located in the same directory as this __init__.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Search patterns for different OSs
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
            raise FileNotFoundError(
                f"Could not find compiled C extension '_capi' in {current_dir}. "
                "Did you run 'pip install .'?"
            )

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

    def solve(self, compute, durations, sizes, N, deadline):

        """
        Solves the transmission scheduling problem using AVX2/Scalar backend.
        
        Parameters
        ----------
        compute : (T,) float64 array
        durations : (T, k) float64 array (in Milliseconds!)
        sizes : (T, k) float64 array
        N : int (Buffer size)
        deadline : float (in Milliseconds!)
        """

        # 1. Setup Standard Inputs
        compute = np.ascontiguousarray(compute, dtype=np.float64)
        durations = np.ascontiguousarray(durations, dtype=np.float64)
        sizes = np.ascontiguousarray(sizes, dtype=np.float64)
        
        # Initial Arrivals (Pure Compute)
        arrivals = np.cumsum(compute)
        # We need to pass 'compute' to C, but C recalculates arrivals internally based on compute.
        # To force "Delayed Arrivals" in C, we must modify the 'compute' array passed to it.
        # Compute[i] = Arrival[i] - Arrival[i-1]
        
        # Constraint: Task i+N cannot start until Task i finishes.
        
        current_compute = compute.copy()
        best_val = 0.0
        best_choices = np.zeros(len(compute), dtype=np.int32)
        
        # Iteration Loop (2-3 passes is usually enough)
        for pass_idx in range(3):
            # Run Solver
            val, choices = self._run_c_solver(current_compute, durations, sizes, N, deadline)
            
            if val == 0.0:
                return 0.0, choices # Impossible
            
            # Calculate actual finish times for this schedule
            # Reconstruct timeline in Python to check buffer constraints
            finish_times = np.zeros(len(compute))
            # Calculate raw arrivals from current_compute
            raw_arrivals = np.cumsum(current_compute)
            
            current_time = 0.0
            for i in range(len(compute)):
                # Task i ready at raw_arrival
                arrival_time = raw_arrivals[i]
                
                # Wait for previous task to finish transmission
                start_time = max(current_time, arrival_time)
                
                # Check Buffer Constraint:
                # Task i cannot enter buffer until Task i-N has finished.
                # So effective_arrival = max(arrival_time, Finish[i-N])
                if i >= N:
                    buffer_open_time = finish_times[i-N]
                    if buffer_open_time > start_time:
                         # We have a bottleneck!
                         start_time = buffer_open_time
                
                # Transmission
                dur = durations[i, choices[i]]
                finish_times[i] = start_time + dur
                current_time = finish_times[i]
            
            # Check if we converged
            # If the calculated start times implied by the buffer match the inputs, we are good.
            # Update 'current_compute' to reflect the forced delays
            
            new_arrivals = np.zeros(len(compute))
            # Base arrivals
            base_arrivals = np.cumsum(compute) 
            
            max_delay = 0.0
            for i in range(len(compute)):
                eff_arrival = base_arrivals[i]
                if i >= N:
                    # The constraint: Arrival[i] >= Finish[i-N]
                    if finish_times[i-N] > eff_arrival:
                        eff_arrival = finish_times[i-N]
                new_arrivals[i] = eff_arrival
            
            # Convert absolute arrivals back to relative 'compute' deltas for the C solver
            new_compute = np.zeros(len(compute))
            new_compute[0] = new_arrivals[0]
            new_compute[1:] = np.diff(new_arrivals)
            
            # Check convergence (if inputs didn't change much)
            if np.allclose(current_compute, new_compute, atol=1e-3):
                best_val = val
                best_choices = choices
                break
                
            current_compute = new_compute
            best_val = val
            best_choices = choices

        return best_val, best_choices

    def _run_c_solver(self, compute, durations, sizes, N, deadline):
        # Helper to call the raw C function
        T, k = durations.shape
        flat_durs = durations.flatten()
        flat_sizes = sizes.flatten()
        choices = np.zeros(T, dtype=np.int32)
        val = self.lib.solve_scheduler(
            T, N, k, compute, flat_durs, flat_sizes, deadline, choices
        )
        return val, choices