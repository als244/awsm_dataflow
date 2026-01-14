import ctypes
import numpy as np
import os
import glob
import sys

class TransmissionScheduler:
    def __init__(self):
        # 1. Find the compiled shared library
        # It will be named something like '_capi.cpython-39-x86_64-linux-gnu.so'
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

    def solve(self, compute, durations, sizes, N):
        """
        Solves the transmission scheduling problem with STRICT Hardware Constraints.
        
        The solver assumes the C backend enforces that Task `i` must finish 
        before Task `i + N` arrives.

        Parameters
        ----------
        compute : array_like of float
            (T,) array. Arrival interval between tasks.
        durations : array_like of float
            (T, k) array. Transmission duration in ms.
        sizes : array_like of float
            (T, k) array. Utility/Size gained by selecting an option.
        N : int
            Buffer hardware limit.

        Returns
        -------
        best_val : float or None
            Max total size. Returns `None` if constraints cannot be met (e.g. buffer overflow).
        best_choices : ndarray or None
            Selected options. Returns `None` on failure.
        """
        # 1. Validate and Cast Inputs
        compute = np.ascontiguousarray(compute, dtype=np.float64)
        durations = np.ascontiguousarray(durations, dtype=np.float64)
        sizes = np.ascontiguousarray(sizes, dtype=np.float64)
        
        T, k = durations.shape
        
        # 2. Calculate "Safe Infinite" Deadline
        # Since we removed the user-facing deadline, we pass a value so large
        # that it is physically impossible to exceed unless the hardware buffer breaks.
        # Safe = Total Arrival Time + Sum of Worst-Case Durations + Padding
        safe_deadline = np.sum(compute) + np.sum(np.max(durations, axis=1)) + 1000.0

        # 3. Handle Negative Scores (Robustness Fix)
        # Even in strict mode, if you pass negative sizes (e.g. for minimization),
        # we must shift them to be positive for the C solver to work correctly.
        min_val = np.min(sizes)
        offset = 0.0
        if min_val <= 0:
            offset = abs(min_val) + 1.0
        
        shifted_sizes = sizes + offset
        if offset > 0:
            shifted_sizes = np.ascontiguousarray(shifted_sizes, dtype=np.float64)
        else:
            shifted_sizes = sizes # No copy needed
            
        # 4. Run C Solver
        choices = np.zeros(T, dtype=np.int32)
        
        # Flatten for C
        flat_durs = durations.flatten()
        flat_sizes = shifted_sizes.flatten()
        
        raw_val = self.lib.solve_scheduler(
            T, N, k, compute, flat_durs, flat_sizes, safe_deadline, choices
        )
        
        # 5. Check Failure vs. Success
        # Since we added `offset` (>= 1.0), any valid schedule MUST have a score >= T * 1.0.
        # Therefore, raw_val == 0.0 implies the C solver found NO path.
        if raw_val == 0.0:
            return None, None
            
        # 6. Restore Real Value
        real_val = raw_val - (T * offset)
            
        return real_val, choices