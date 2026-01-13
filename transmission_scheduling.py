import numpy as np
from typing import List, Tuple
import heapq

def solve_transmission_scheduling(
    compute_times: List[float],
    transmission_options: List[List[Tuple[float, float]]],  # For each task: list of (size, time) pairs
    N: int
) -> Tuple[float, List[int]]:
    """
    Solve the transmission scheduling problem.
    
    Args:
        compute_times: t_i for each task (length T)
        transmission_options: For each task, list of (size, transmission_time) pairs
        N: Buffer count / window size
    
    Returns:
        (total_size, choices) where choices[i] is the index of chosen option for task i
    
    Constraint: Transmission for task i must complete before task i+N starts.
    Transmissions are sequential (FIFO queue).
    """
    T = len(compute_times)
    
    # Precompute prefix sums of compute times for fast range queries
    # C[i] = t_0 + t_1 + ... + t_{i-1}  (C[0] = 0)
    C = [0.0] * (T + N + 1)
    for i in range(T):
        C[i + 1] = C[i] + compute_times[i]
    # Extend with zeros for tasks beyond T (or we could handle boundary specially)
    for i in range(T, T + N):
        C[i + 1] = C[i]
    
    # DP approach:
    # State: cumulative transmission time X after choosing for task i
    # Constraint: X <= C[i + N] for task i's transmission to finish in time
    #
    # Since X is continuous, we discretize or use a different approach.
    # 
    # Key insight: We can use DP where state is the "slack" at each position.
    # slack_i = C[i + N] - X_i (how much buffer we have)
    #
    # But slack depends on future compute times which we know!
    # 
    # Let's track X directly and use the constraint X <= C[i + N]
    
    # For efficiency with continuous state, we note:
    # - At each step, we have a set of Pareto-optimal (X, total_size) pairs
    # - We prune dominated states
    # 
    # This works because if state A has X_A <= X_B and size_A >= size_B, 
    # then A dominates B.
    
    # State: list of (cumulative_transmission_time, total_size, choices)
    # We keep Pareto frontier on (X, -size) — minimize X, maximize size
    
    # Initial state: no transmissions yet
    states = [(0.0, 0.0, [])]  # (X, total_size, choices)
    
    for i in range(T):
        deadline = C[i + N]  # Transmission for task i must complete by this time
        
        new_states = []
        for X, total_size, choices in states:
            for opt_idx, (size, trans_time) in enumerate(transmission_options[i]):
                new_X = X + trans_time
                if new_X <= deadline + 1e-9:  # Feasible
                    new_states.append((new_X, total_size + size, choices + [opt_idx]))
        
        # Prune dominated states: keep Pareto frontier
        # Sort by X ascending, then by size descending
        new_states.sort(key=lambda s: (s[0], -s[1]))
        
        pruned = []
        best_size = -1
        for X, size, choices in new_states:
            if size > best_size:
                pruned.append((X, size, choices))
                best_size = size
        
        states = pruned
        
        if not states:
            raise ValueError(f"No feasible solution at task {i}")
    
    # Return best solution (maximum size)
    best = max(states, key=lambda s: s[1])
    return best[1], best[2]


def solve_transmission_scheduling_fast(
    compute_times: List[float],
    transmission_options: List[List[Tuple[float, float]]],
    N: int
) -> Tuple[float, List[int]]:
    """
    Faster version using the insight that we only need to track Pareto-optimal states.
    
    The number of Pareto-optimal states is bounded by O(T * k) in practice,
    making this feasible for large inputs.
    """
    T = len(compute_times)
    
    # Precompute prefix sums
    C = [0.0] * (T + N + 1)
    for i in range(T):
        C[i + 1] = C[i] + compute_times[i]
    for i in range(T, T + N):
        C[i + 1] = C[i]
    
    # State: dict mapping X -> (best_size_for_this_X, choices)
    # We'll maintain Pareto frontier more efficiently
    
    # Use a list of (X, size, choice_history) sorted by X
    # Pareto frontier: as X increases, size must also increase (otherwise dominated)
    
    states = {0.0: (0.0, [])}  # X -> (total_size, choices)
    
    for i in range(T):
        deadline = C[i + N]
        
        new_states = {}
        
        for X, (total_size, choices) in states.items():
            for opt_idx, (size, trans_time) in enumerate(transmission_options[i]):
                new_X = X + trans_time
                if new_X <= deadline + 1e-9:
                    new_size = total_size + size
                    # Round X to avoid floating point explosion of states
                    new_X_rounded = round(new_X, 9)
                    if new_X_rounded not in new_states or new_states[new_X_rounded][0] < new_size:
                        new_states[new_X_rounded] = (new_size, choices + [opt_idx])
        
        # Build Pareto frontier
        sorted_states = sorted(new_states.items(), key=lambda x: x[0])
        pruned = {}
        best_size = -1
        for X, (size, choices) in sorted_states:
            if size > best_size:
                pruned[X] = (size, choices)
                best_size = size
        
        states = pruned
        
        if not states:
            raise ValueError(f"No feasible solution at task {i}")
    
    # Find best
    best_X, (best_size, best_choices) = max(states.items(), key=lambda x: x[1][0])
    return best_size, best_choices


def solve_fast_no_history(
    compute_times: List[float],
    transmission_options: List[List[Tuple[float, float]]],
    N: int,
    final_deadline: float = None
) -> Tuple[float, List[int]]:
    """
    Memory-efficient version that reconstructs choices at the end.
    """
    total_size, choices, _ = solve_fast_no_history_with_stats(
        compute_times, transmission_options, N, final_deadline
    )
    return total_size, choices


def solve_fast_no_history_with_stats(
    compute_times: List[float],
    transmission_options: List[List[Tuple[float, float]]],
    N: int,
    final_deadline: float = None
) -> Tuple[float, List[int], List[int]]:
    """
    Memory-efficient version that reconstructs choices at the end.
    Also returns frontier sizes for analysis.
    
    Args:
        compute_times: t_i for each task
        transmission_options: For each task, list of (size, transmission_time) pairs
        N: Buffer count / window size
        final_deadline: Optional absolute deadline by which ALL transmissions must complete.
                       If None, only the sliding window constraints apply.
    
    Constraints:
        1. Transmission for task i must complete before task i+N starts (sliding window)
        2. All transmissions must complete by final_deadline (if specified)
    """
    T = len(compute_times)
    
    # Precompute prefix sums
    # C[i] = t_0 + t_1 + ... + t_{i-1} = time when task i-1 completes
    C = [0.0] * (T + N + 1)
    for i in range(T):
        C[i + 1] = C[i] + compute_times[i]
    for i in range(T, T + N):
        C[i + 1] = C[i]
    
    # If final_deadline not specified, default to no additional constraint
    # (set it to infinity effectively)
    if final_deadline is None:
        final_deadline = float('inf')
    
    # Forward pass: compute Pareto frontiers
    # Store each frontier for backtracking
    frontiers = []
    frontier_sizes = []
    
    # frontier: list of (X, size) tuples, sorted by X, Pareto-optimal
    frontier = [(0.0, 0.0)]
    frontiers.append(frontier)
    frontier_sizes.append(len(frontier))
    
    for i in range(T):
        # Sliding window constraint: must finish before task i+N starts
        window_deadline = C[i + N]
        
        # Final deadline constraint: all transmissions must finish by final_deadline
        # For task i, this means cumulative transmission X_{i+1} <= final_deadline
        
        # The effective deadline is the minimum of both constraints
        deadline = min(window_deadline, final_deadline)
        
        new_points = []
        for X, total_size in frontier:
            for opt_idx, (size, trans_time) in enumerate(transmission_options[i]):
                new_X = X + trans_time
                if new_X <= deadline + 1e-9:
                    new_points.append((round(new_X, 9), total_size + size))
        
        # Build Pareto frontier
        new_points.sort(key=lambda p: (p[0], -p[1]))
        new_frontier = []
        best_size = -1
        for X, size in new_points:
            if size > best_size:
                new_frontier.append((X, size))
                best_size = size
        
        frontier = new_frontier
        frontiers.append(frontier)
        frontier_sizes.append(len(frontier))
        
        if not frontier:
            raise ValueError(f"No feasible solution at task {i}")
    
    # Backtrack to find choices
    choices = []
    target_size = max(p[1] for p in frontiers[T])
    target_X = None
    for X, size in frontiers[T]:
        if abs(size - target_size) < 1e-9:
            target_X = X
            break
    
    for i in range(T - 1, -1, -1):
        # Find which choice led to (target_X, target_size)
        found = False
        for prev_X, prev_size in frontiers[i]:
            for opt_idx, (size, trans_time) in enumerate(transmission_options[i]):
                new_X = round(prev_X + trans_time, 9)
                new_size = prev_size + size
                if abs(new_X - target_X) < 1e-9 and abs(new_size - target_size) < 1e-9:
                    choices.append(opt_idx)
                    target_X = prev_X
                    target_size = prev_size
                    found = True
                    break
            if found:
                break
        if not found:
            raise ValueError(f"Backtracking failed at task {i}")
    
    choices.reverse()
    return max(p[1] for p in frontiers[T]), choices, frontier_sizes


# ============ TEST CASES ============

def test_small():
    """Small test case we can verify by hand."""
    print("=" * 50)
    print("TEST: Small case")
    print("=" * 50)
    
    # 4 tasks, N=2 buffers
    # Task i's transmission must complete before task i+2 starts
    compute_times = [10, 10, 10, 10]
    
    # Options: (size, transmission_time)
    # Bigger size = more time
    transmission_options = [
        [(1, 5), (2, 10), (3, 15), (4, 20)],   # Task 0
        [(1, 5), (2, 10), (3, 15), (4, 20)],   # Task 1
        [(1, 5), (2, 10), (3, 15), (4, 20)],   # Task 2
        [(1, 5), (2, 10), (3, 15), (4, 20)],   # Task 3
    ]
    N = 2
    
    # Deadline for task 0: C[0+2] = t_0 + t_1 = 20
    # Deadline for task 1: C[1+2] = t_0 + t_1 + t_2 = 30
    # etc.
    #
    # τ_0 <= 20 (can pick up to size 2, time 10... wait let's recalc)
    # After task 0 completes at time t_0=10, we start transmitting.
    # Must finish before task 2 starts at time t_0 + t_1 = 20.
    # So τ_0 <= 10. Can pick size 2.
    #
    # After task 1 completes at time 20, transmit.
    # Must finish before task 3 starts at time 30.
    # But τ_0 might still be going! 
    # Total transmission by time 30: τ_0 + τ_1 <= 30 - 10 = 20
    # If τ_0 = 10, then τ_1 <= 10. Can pick size 2 again.
    
    total_size, choices = solve_fast_no_history(compute_times, transmission_options, N)
    
    print(f"Compute times: {compute_times}")
    print(f"N = {N}")
    print(f"Optimal total size: {total_size}")
    print(f"Choices: {choices}")
    chosen_sizes = [transmission_options[i][c][0] for i, c in enumerate(choices)]
    chosen_times = [transmission_options[i][c][1] for i, c in enumerate(choices)]
    print(f"Chosen sizes: {chosen_sizes}")
    print(f"Chosen transmission times: {chosen_times}")
    
    # Verify feasibility
    verify_solution(compute_times, transmission_options, N, choices)


def test_medium():
    """Medium test with varied compute times."""
    print("\n" + "=" * 50)
    print("TEST: Medium case")
    print("=" * 50)
    
    np.random.seed(42)
    T = 20
    N = 5
    
    compute_times = list(np.random.uniform(5, 20, T))
    
    # Same options for all tasks
    base_options = [(1, 2), (2, 5), (3, 10), (4, 18)]
    transmission_options = [base_options.copy() for _ in range(T)]
    
    total_size, choices = solve_fast_no_history(compute_times, transmission_options, N)
    
    print(f"T = {T}, N = {N}")
    print(f"Optimal total size: {total_size}")
    print(f"Choices: {choices}")
    
    verify_solution(compute_times, transmission_options, N, choices)


def test_large():
    """Large test matching the original parameters: k=4, N=100, T=1000."""
    print("\n" + "=" * 50)
    print("TEST: Large case (T=1000, N=100, k=4)")
    print("=" * 50)
    
    np.random.seed(123)
    T = 1000
    N = 100
    
    compute_times = list(np.random.uniform(10, 50, T))
    
    # k=4 options, transmission time roughly proportional to size
    base_options = [(1, 5), (2, 12), (3, 22), (4, 35)]
    transmission_options = [base_options.copy() for _ in range(T)]
    
    import time
    start = time.time()
    total_size, choices, frontier_sizes = solve_fast_no_history_with_stats(compute_times, transmission_options, N)
    elapsed = time.time() - start
    
    print(f"Optimal total size: {total_size}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"Max frontier size: {max(frontier_sizes)}")
    print(f"Avg frontier size: {np.mean(frontier_sizes):.1f}")
    print(f"Choice distribution: {np.bincount(choices)}")
    
    verify_solution(compute_times, transmission_options, N, choices)


def test_tight_constraints():
    """Test where constraints are very tight."""
    print("\n" + "=" * 50)
    print("TEST: Tight constraints")
    print("=" * 50)
    
    # Short compute times, long transmission times
    T = 10
    N = 3
    compute_times = [5] * T  # All same
    
    # Options where even smallest takes significant time
    transmission_options = [
        [(1, 4), (2, 8), (3, 12), (4, 16)]
        for _ in range(T)
    ]
    
    # Window of N=3 gives us 3*5=15 time units per window
    # If we always pick size 1 (time 4), total τ over 3 tasks = 12 <= 15 ✓
    # If we try size 2 (time 8), total τ over 3 tasks = 24 > 15 ✗
    # So we should mostly pick size 1, maybe occasionally size 2
    
    total_size, choices = solve_fast_no_history(compute_times, transmission_options, N)
    
    print(f"Compute times: {compute_times}")
    print(f"N = {N}")
    print(f"Optimal total size: {total_size}")
    print(f"Choices: {choices}")
    
    verify_solution(compute_times, transmission_options, N, choices)


def test_your_parameters():
    """Test case with N=10, k=4, T=64 as requested, with final deadline."""
    print("\n" + "=" * 50)
    print("TEST: Your parameters (T=64, N=10, k=4) with final deadline")
    print("=" * 50)
    
    np.random.seed(999)
    T = 64
    N = 10
    k = 4
    
    # Compute times between 10-30 time units
    compute_times = list(np.random.uniform(10, 30, T))
    total_compute_time = sum(compute_times)
    
    # k=4 options: (size, transmission_time)
    # Larger sizes take proportionally more time
    base_options = [(1, 3), (2, 7), (3, 12), (4, 20)]
    transmission_options = [base_options.copy() for _ in range(T)]
    
    print(f"Total compute time: {total_compute_time:.1f}")
    print(f"If all tasks pick max size (time=20): total trans = {T * 20}")
    print(f"If all tasks pick min size (time=3): total trans = {T * 3}")
    
    # Test 1: No final deadline (only sliding window constraint)
    print("\n--- Without final deadline ---")
    total_size, choices = solve_fast_no_history(compute_times, transmission_options, N)
    total_trans = sum(transmission_options[i][choices[i]][1] for i in range(T))
    
    print(f"Optimal total size: {total_size}")
    print(f"Total transmission time used: {total_trans:.1f}")
    print(f"Choice distribution: {np.bincount(choices, minlength=k)}")
    verify_solution(compute_times, transmission_options, N, choices)
    
    # Test 2: With a tight final deadline
    # Set deadline to be somewhat constraining
    final_deadline = total_compute_time * 0.6  # 60% of total compute time
    
    print(f"\n--- With final deadline = {final_deadline:.1f} (60% of compute time) ---")
    try:
        total_size_constrained, choices_constrained = solve_fast_no_history(
            compute_times, transmission_options, N, final_deadline
        )
        total_trans_constrained = sum(
            transmission_options[i][choices_constrained[i]][1] for i in range(T)
        )
        
        print(f"Optimal total size: {total_size_constrained}")
        print(f"Total transmission time used: {total_trans_constrained:.1f}")
        print(f"Choice distribution: {np.bincount(choices_constrained, minlength=k)}")
        verify_solution(compute_times, transmission_options, N, choices_constrained, final_deadline)
    except ValueError as e:
        print(f"No feasible solution: {e}")
    
    # Test 3: With a very tight final deadline
    final_deadline_tight = total_compute_time * 0.3  # 30% of total compute time
    
    print(f"\n--- With tight final deadline = {final_deadline_tight:.1f} (30% of compute time) ---")
    try:
        total_size_tight, choices_tight = solve_fast_no_history(
            compute_times, transmission_options, N, final_deadline_tight
        )
        total_trans_tight = sum(
            transmission_options[i][choices_tight[i]][1] for i in range(T)
        )
        
        print(f"Optimal total size: {total_size_tight}")
        print(f"Total transmission time used: {total_trans_tight:.1f}")
        print(f"Choice distribution: {np.bincount(choices_tight, minlength=k)}")
        verify_solution(compute_times, transmission_options, N, choices_tight, final_deadline_tight)
    except ValueError as e:
        print(f"No feasible solution: {e}")


def verify_solution(compute_times, transmission_options, N, choices, final_deadline=None):
    """Verify that a solution satisfies all constraints."""
    T = len(compute_times)
    
    # Compute prefix sums
    C = [0.0]
    for t in compute_times:
        C.append(C[-1] + t)
    # Pad for tasks beyond T
    for _ in range(N):
        C.append(C[-1])
    
    # Compute transmission times
    trans_times = [transmission_options[i][choices[i]][1] for i in range(T)]
    
    # Cumulative transmission time
    X = [0.0]
    for τ in trans_times:
        X.append(X[-1] + τ)
    
    # Check sliding window constraints: X[i+1] <= C[i+N] for all i
    all_ok = True
    for i in range(T):
        deadline = C[i + N]
        actual = X[i + 1]
        if actual > deadline + 1e-9:
            print(f"  VIOLATION (window) at task {i}: X[{i+1}]={actual:.2f} > C[{i+N}]={deadline:.2f}")
            all_ok = False
    
    # Check final deadline constraint
    if final_deadline is not None:
        total_trans_time = X[T]
        if total_trans_time > final_deadline + 1e-9:
            print(f"  VIOLATION (final deadline): total transmission time {total_trans_time:.2f} > {final_deadline:.2f}")
            all_ok = False
        else:
            print(f"  Final transmission time: {total_trans_time:.2f} <= {final_deadline:.2f} ✓")
    
    if all_ok:
        print("  ✓ All constraints satisfied!")
    
    return all_ok


if __name__ == "__main__":
    test_small()
    test_medium()
    test_large()
    test_tight_constraints()
    test_your_parameters()