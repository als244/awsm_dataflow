import numpy as np
from transmission_scheduler import TransmissionScheduler

# Init (loads the C library automatically)
sched = TransmissionScheduler()

# Dummy Data
T, N, k = 64, 10, 4
compute = np.ones(T) * 10.0
durs = np.ones((T, k)) * 100.0  # 100ms
sizes = durs * 5.0
deadline = 20000.0

# Solve
max_val, choices = sched.solve(compute, durs, sizes, N, deadline)

print(f"Result: {max_val}")
print(f"Choices: {choices}")