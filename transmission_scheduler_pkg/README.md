# Transmission Scheduler

A high-performance solver for buffer-constrained transmission scheduling. Given a sequence of tasks that produce data, the solver selects the optimal transmission option for each task to maximize total transmitted data while respecting hardware buffer limits.

## Installation

```bash
cd transmission_scheduler_pkg
pip install -e .
```

Requires a C compiler with optimization support. AVX2 is auto-detected at runtime and used when available.

## Python API

The package exposes a single class:

```python
from transmission_scheduler import TransmissionScheduler

ts = TransmissionScheduler()
best_val, choices = ts.solve(compute, durations, sizes, N)
```

### `TransmissionScheduler.solve(compute, durations, sizes, N)`

Solves the transmission scheduling problem: for each of `T` sequential tasks, select one of `k` transmission options to maximize the total transmitted size, subject to a hardware buffer constraint.

**Parameters:**

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `compute` | `array_like[float]` | `(T,)` | Inter-arrival time between consecutive tasks, in milliseconds. Task `i` arrives at cumulative time `sum(compute[:i+1])`. |
| `durations` | `array_like[float]` | `(T, k)` | Transmission duration for each option, in milliseconds. `durations[i, j]` is the time to transmit task `i` using option `j`. |
| `sizes` | `array_like[float]` | `(T, k)` | Value (e.g. data size) gained by selecting each option. `sizes[i, j]` is the reward for transmitting task `i` with option `j`. Can be zero or negative. |
| `N` | `int` | -- | Hardware buffer depth. Task `i` must finish transmitting before task `i + N - 1` arrives, otherwise the buffer overflows. |

**Returns:**

| Name | Type | Description |
|------|------|-------------|
| `best_val` | `float` or `None` | Maximum total size across all tasks. `None` if no feasible schedule exists. |
| `choices` | `ndarray[int32]` or `None` | Array of length `T` where `choices[i]` is the selected option index (0 to k-1) for task `i`. `None` on failure. |

**Example:**

```python
import numpy as np
from transmission_scheduler import TransmissionScheduler

ts = TransmissionScheduler()

compute = np.array([10.0, 10.0, 10.0])            # 3 tasks, 10ms apart
durations = np.array([[2.0, 5.0], [2.0, 5.0], [2.0, 5.0]])  # 2 options each
sizes = np.array([[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]])      # option 1 is bigger

val, choices = ts.solve(compute, durations, sizes, N=2)
# val = 9.0, choices = [1, 1, 1]  (picks the bigger option every time)
```

### Behavior notes

- **Negative sizes**: Handled automatically. The wrapper shifts all values to be strictly positive before calling the C solver, then corrects the returned objective value.
- **Failure**: Returns `(None, None)` when no feasible schedule exists (i.e., the buffer constraint cannot be satisfied for any combination of options).
- **Time precision**: The solver discretizes time at 0.1ms resolution (10 ticks per millisecond).
- **Buffer safety**: The sliding window buffer is sized dynamically from the input data. There is no hardcoded limit that can be outgrown, so the solver is guaranteed not to silently clip DP states regardless of input size, task count, or buffer depth `N`.

## How the solver works

### Problem formulation

The solver maximizes:

```
maximize  sum_{i=0}^{T-1} sizes[i, choices[i]]
```

subject to:

1. **Sequential transmission**: Transmissions do not overlap. Task `i` cannot start transmitting until task `i-1` finishes.
2. **Arrival constraint**: Task `i` cannot start transmitting before it arrives (at cumulative time `sum(compute[:i+1])`).
3. **Buffer constraint**: Task `i` must finish transmitting before task `i + N - 1` arrives. This prevents the hardware buffer (depth `N`) from overflowing.

If a task arrives and the transmitter is idle (finished the previous task early), the task waits until arrival and then starts immediately ("wait logic"). If the transmitter is still busy, the task starts as soon as the previous transmission finishes ("pull logic").

### Algorithm: Sliding-window dynamic programming

The solver uses a 1D dynamic programming approach where the state is the finish time of the most recent transmission, discretized into integer ticks.

**State**: `dp[t]` = maximum total size achievable when the most recent transmission finishes at time `t`.

**Transitions** (for each task `i`, for each option `j`):

- **Wait**: If the transmitter finishes at any time `t <= arrival[i]`, task `i` waits and starts at `arrival[i]`. Finish time = `arrival[i] + duration[i][j]`. Value = `max(dp[0..arrival[i]]) + size[i][j]`.
- **Pull**: If the transmitter finishes at time `t > arrival[i]`, task `i` starts immediately at `t`. Finish time = `t + duration[i][j]`. Value = `dp[t] + size[i][j]`.

Both transitions are rejected if the finish time exceeds `deadline[i] = arrival[i + N - 1]` (the buffer constraint).

**Sliding window**: The active DP band (non-negative-infinity entries) is typically only a few hundred ticks wide, even though absolute times can span millions of ticks over a long task sequence. Instead of allocating an array covering the full timeline, the solver maintains a small window (`buf_size` ticks) and periodically re-centers it by shifting the array contents so the active band stays within bounds.

The window size is computed dynamically by `compute_buf_size()`: it simulates the DP band evolution across all tasks to find the worst-case band width, applies a 2x safety margin, and rounds up to a power of 2. This guarantees no states are silently clipped, regardless of input data.

**Traceback**: For each task and finish-time tick, the solver records a `TraceCell` containing the chosen option and the absolute time of the predecessor state. After the forward pass finds the optimal value, it walks backward through these records to reconstruct the full choice sequence. The per-task window offset is stored alongside the history so that absolute times can be converted to relative window indices during traceback.

### Complexity

- **Time**: O(T * k * W) where W is the active band width (typically a few hundred ticks).
- **Space**: O(T * buf_size) for the traceback history. `buf_size` is data-dependent but typically small (e.g. 2048 for the test data, yielding ~34MB for T=2048).

## File structure

```
transmission_scheduler_pkg/
  setup.py                          # Build configuration (setuptools + C extension)
  README.md                         # This file
  transmission_scheduler/
    __init__.py                     # Python API: TransmissionScheduler class
    transmission_scheduler.h        # C header: solve_scheduler() + compute_buf_size()
    transmission_scheduler.c        # C implementation: scalar + AVX2 solvers
```

### `__init__.py` -- Python wrapper

Loads the compiled C shared library via `ctypes` and provides the `TransmissionScheduler` class. Responsibilities:

- Input validation and type casting (ensures contiguous float64 arrays).
- Computing the safe deadline (an upper bound on the latest possible finish time, used as a fallback when the buffer constraint does not apply to the last few tasks).
- Calling `compute_buf_size()` in C to determine the sliding window size from the data.
- Handling negative sizes by shifting all values positive before calling C, then correcting the result afterward.
- Translating C return values: `0.0` from the C solver means failure, mapped to `(None, None)`.

### `transmission_scheduler.h` -- C header

Declares two public C functions:

- **`solve_scheduler()`**: The main solver. Accepts all problem parameters plus `time_scale` (ticks per millisecond), `buf_size` (sliding window width), and a caller-allocated output array.
- **`compute_buf_size()`**: Computes the minimum safe `buf_size` for a given problem instance by simulating the DP band evolution. Returns a power-of-2 value with 2x headroom.

### `transmission_scheduler.c` -- C solver

Contains two implementations of the same sliding-window DP algorithm, plus the `compute_buf_size()` helper:

- **`solve_scalar()`**: Portable implementation using standard C. Used as a fallback on systems without AVX2.
- **`solve_avx2()`**: Optimized implementation for AVX2-capable x86_64 systems, auto-selected at runtime via `__builtin_cpu_supports("avx2")`. The pull loop uses AVX2 intrinsics (`_mm256_add_pd`, `_mm256_max_pd`) to compute DP value updates 4 cells at a time. Traceback recording uses a mask-based approach: after each AVX max, `_mm256_cmp_pd` + `_mm256_movemask_pd` identifies which lanes improved, and `TraceCell` writes happen only for those lanes. The speedup over scalar scales with the active band width — modest (~5%) for narrow bands (~150 ticks), more significant for wider bands.
- **`compute_buf_size()`**: Simulates the DP band evolution (tracking both `sim_min` and `sim_max` per step, clipped by deadlines) to find the worst-case buffer span — the maximum distance from the earliest active state to the furthest reachable finish time. Adds padding and rounds up to a power of 2 (minimum 2048).

Both solvers share the same structure:

1. **Precompute** arrival times (cumulative sum of `compute * time_scale`) and per-task deadlines from the buffer constraint.
2. **Forward DP pass** with sliding window re-centering: when the relative arrival index reaches `buf_size / 2`, the active band is shifted to the start of the array via `memmove`.
3. **Traceback** to reconstruct the optimal choice sequence using per-cell `TraceCell` records (option index + absolute predecessor time).

All buffer sizes are runtime parameters -- there are no hardcoded limits that constrain problem size.

Key compile-time constants:

| Constant | Value | Purpose |
|----------|-------|---------|
| `INF_NEG` | -1e18 | Sentinel for uninitialized DP cells. |
