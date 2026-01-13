import ctypes
import numpy as np
import os
import platform

class TransmissionScheduler:
    def __init__(self, lib_path=None):
        if lib_path is None:
            lib_name = "libtransmission_scheduler.dylib" if platform.system() == "Darwin" else "libtransmission_scheduler.so"
            lib_path = os.path.abspath(lib_name)

        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Library not found at {lib_path}. Run ./build.sh")

        self.lib = ctypes.CDLL(lib_path)
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
        compute = np.ascontiguousarray(compute, dtype=np.float64)
        durations = np.ascontiguousarray(durations, dtype=np.float64).flatten()
        sizes = np.ascontiguousarray(sizes, dtype=np.float64).flatten()
        choices = np.zeros(len(compute), dtype=np.int32)
        
        val = self.lib.solve_scheduler(
            len(compute), N, len(durations)//len(compute), 
            compute, durations, sizes, deadline, choices
        )
        return val, choices