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

    def solve(self, compute, durations, sizes, N, deadline):
        """
        Solves the transmission scheduling problem using an iterative 'Dynamic Alpha' approach.
        
        This method attempts to maximize total 'size' while minimizing source idle time caused 
        by buffer congestion. It runs multiple passes of the underlying C solver. If a task 
        blocks the buffer (causing downstream tasks to wait), it is penalized in subsequent 
        passes to encourage faster processing.

        Parameters
        ----------
        compute : array_like of float
            A 1D array of shape `(T,)` representing the fixed compute time (arrival interval) 
            between tasks. Task `i` arrives `compute[i]` ms after Task `i-1`.
        
        durations : array_like of float
            A 2D array of shape `(T, k)` representing the transmission duration (in ms) 
            for each of the `k` options for every task.
        
        sizes : array_like of float
            A 2D array of shape `(T, k)` representing the utility/size gained by selecting 
            a specific option.
        
        N : int
            The buffer lookahead constraint. Task `i` must finish transmission before 
            Task `i + N` can enter the buffer.
        
        deadline : float
            The global deadline (in ms) by which the *last* task must finish. 
            If deadline is loose, set to `float('inf')`.

        Returns
        -------
        best_val : float
            The maximum total size achievable from the best schedule found. 
            Returns `0.0` if the C solver fails to find any valid path.
        
        best_choices : ndarray of int32
            A 1D array of shape `(T,)` containing the selected option index `[0, k)` 
            for each task.
            
        idle_times : ndarray of float64
            A 1D array of shape `(T,)` representing the "Source Idle Time" (or Wait Time) 
            for each task.
            - `idle_times[i] > 0` means Task `i` arrived but had to wait for the buffer 
              to open (because Task `i-N` was still transmitting).
            - The first `N` tasks will always have `0.0` idle time.

        """
        # 1. Validate and Cast Inputs
        compute = np.ascontiguousarray(compute, dtype=np.float64)
        durations = np.ascontiguousarray(durations, dtype=np.float64)
        sizes = np.ascontiguousarray(sizes, dtype=np.float64)
        
        # Initialize Dynamic Alphas (Starts at 0 - Greedy Strategy)
        current_alphas = np.zeros(len(compute)) 
        
        best_val = 0.0
        best_choices = np.zeros(len(compute), dtype=np.int32)
        best_idle_times = np.zeros(len(compute))
        
        # 2. Iteration Loop (Smart Feedback)
        # We iterate up to 5 times to resolve bottlenecks.
        for pass_idx in range(5): 
            
            # A. Apply Alphas locally to create "Effective Sizes"
            # Effective Size = Size - (Duration * Alpha[i])
            # We only modify sizes where alpha > 0 (bottlenecks)
            effective_sizes = sizes.copy()
            for t in range(len(compute)):
                if current_alphas[t] > 0:
                    effective_sizes[t, :] -= (durations[t, :] * current_alphas[t])
            
            effective_sizes = np.ascontiguousarray(effective_sizes, dtype=np.float64)

            # B. Run C Solver
            val, choices = self._run_c_solver(compute, durations, effective_sizes, N, deadline)

            if val == 0.0: 
                # If solver fails entirely (e.g., impossible global deadline), return zeros
                return 0.0, choices, np.zeros(len(compute))
            
            # C. Reconstruct Schedule & Measure Idle Time
            finish_times = np.zeros(len(compute))
            raw_arrivals = np.cumsum(compute)
            current_time = 0.0
            
            pass_idle_times = np.zeros(len(compute))
            bottleneck_detected = False
            
            for i in range(len(compute)):
                arrival_time = raw_arrivals[i]
                
                # Earliest physical start (Transmitter free)
                start_time = max(current_time, arrival_time)
                
                # Check Buffer Constraint (The "Idle Time" Source)
                if i >= N:
                    buffer_open_time = finish_times[i-N]
                    
                    # If buffer opens AFTER the task arrived, we have Source Idle Time
                    if buffer_open_time > arrival_time:
                        wait = buffer_open_time - arrival_time
                        pass_idle_times[i] = wait
                        
                        # Identify Bottleneck
                        # If wait is significant (> 1ms), penalize the task (i-N) holding the lock
                        if wait > 1.0: 
                             current_alphas[i-N] += 0.5  # Increase urgency for the blocking task
                             bottleneck_detected = True
                        
                        # The task starts later due to buffer block
                        start_time = max(start_time, buffer_open_time)
                
                # Execute Task
                dur = durations[i, choices[i]]
                finish_times[i] = start_time + dur
                current_time = finish_times[i]

            # Store result of this pass
            best_val = val
            best_choices = choices
            best_idle_times = pass_idle_times
            
            # D. Stop early if no new bottlenecks were found (Converged)
            if not bottleneck_detected:
                break
                
        return best_val, best_choices, best_idle_times

    def _run_c_solver(self, compute, durations, sizes, N, deadline):
        T, k = durations.shape
        flat_durs = durations.flatten()
        flat_sizes = sizes.flatten()
        choices = np.zeros(T, dtype=np.int32)
        val = self.lib.solve_scheduler(
            T, N, k, compute, flat_durs, flat_sizes, deadline, choices
        )
        return val, choices