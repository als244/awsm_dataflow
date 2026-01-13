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
        # Validate and Cast
        compute = np.ascontiguousarray(compute, dtype=np.float64)
        durations = np.ascontiguousarray(durations, dtype=np.float64)
        sizes = np.ascontiguousarray(sizes, dtype=np.float64)

        if len(durations.shape) != 2:
            raise ValueError("Durations must be 2D (T, k)")
            
        T, k = durations.shape
        flat_durs = durations.flatten()
        flat_sizes = sizes.flatten()
        choices = np.zeros(T, dtype=np.int32)
        
        val = self.lib.solve_scheduler(
            T, N, k, 
            compute, flat_durs, flat_sizes, deadline, choices
        )
        return val, choices