import time
import numpy as np
from transmission_scheduler import TransmissionScheduler

def run_test():
    sched = TransmissionScheduler()
    T, N, k = 64, 10, 4
    
    compute = np.random.uniform(8.0, 12.0, size=T)
    # Make k options per task
    durations = np.random.uniform(5.0, 15.0, size=(T, k))
    sizes = durations * 10.0
    
    # Warmup
    sched.solve(compute, durations, sizes, N)
    
    # Bench
    start = time.perf_counter()
    for _ in range(1000):
        sched.solve(compute, durations, sizes, N)
    print(f"Avg Solve Time: {((time.perf_counter()-start)/1000)*1e6:.2f} µs")

    val, choices = sched.solve(compute, durations, sizes, N)
    print(f"Max Size: {val:.2f}")
    print(f"Choices: {choices[:10]}...")

if __name__ == "__main__":
    run_test()