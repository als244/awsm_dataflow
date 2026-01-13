import numpy as np
from transmission_scheduler import TransmissionScheduler

sched = TransmissionScheduler()
T, N, k = 64, 10, 4
compute = np.ones(T) * 100.0

# --- SETUP TRADE-OFFS ---
# Option 0: Slow but High Value (Huge Size)
# Option 3: Fast but Low Value (Small Size)
durs = np.zeros((T, k))
sizes = np.zeros((T, k))

for i in range(T):
    # Option 0: 200ms, Size 1000
    durs[i, 0] = 200.0
    sizes[i, 0] = 1000.0
    
    # Option 1: 150ms, Size 600
    durs[i, 1] = 150.0
    sizes[i, 1] = 600.0
    
    # Option 2: 100ms, Size 300
    durs[i, 2] = 100.0
    sizes[i, 2] = 300.0
    
    # Option 3: 50ms, Size 100
    durs[i, 3] = 50.0
    sizes[i, 3] = 100.0

# Run Solver
max_val, choices = sched.solve(compute, durs, sizes, N)

print(f"Result: {max_val}")
print(f"Choices: {choices}")
