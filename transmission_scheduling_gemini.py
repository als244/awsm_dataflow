import time
import random

def solve_buffer_constrained_schedule(compute_times, N, options, final_deadline, time_resolution=1.0):
    """
    time_resolution: The 'bucket size' for time in ms. 
                     e.g., 1.0 means we round everything to nearest millisecond.
                     Higher = Faster but less accurate. Lower = Slower but exact.
    """
    start_bench = time.perf_counter()
    T = len(compute_times)

    # 1. Precompute Timeline (Same as before)
    arrivals = []
    current_clock = 0.0
    for t in compute_times:
        current_clock += t
        arrivals.append(current_clock)
        
    task_deadlines = []
    for i in range(T):
        if i + N < T:
            d = arrivals[i + N]
        else:
            d = float('inf')
        task_deadlines.append(min(d, final_deadline))

    # 2. QUANTIZED DP
    # Instead of a list of tuples, we use a Dictionary: { quantized_time_int: max_size }
    # This automatically handles "merging" collisions.
    
    # State: { time_bucket_index : max_bits_collected }
    current_states = {0: 0.0}
    
    for i in range(T):
        next_states = {}
        
        arrival = arrivals[i]
        deadline = task_deadlines[i]
        task_opts = options[i]

        if not current_states:
            return 0.0, 0.0

        # We convert the entire 'current_states' dict to items to iterate
        # This loop is now bounded by (TotalTime / resolution) rather than combinatorial explosion
        for prev_time_idx, prev_size in current_states.items():
            
            # Convert index back to real time
            prev_finish_real = prev_time_idx * time_resolution
            
            start_send = max(prev_finish_real, arrival)
            
            for (opt_size, opt_dur) in task_opts:
                finish_send = start_send + opt_dur
                
                if finish_send <= deadline:
                    new_size = prev_size + opt_size
                    
                    # QUANTIZE: Convert new finish time to bucket index
                    # We use ceil to be conservative (ensure we don't violate deadlines by rounding down)
                    # or round() for average case. Let's use int() rounding for speed/simplicity.
                    new_idx = int(math.ceil(finish_send / time_resolution))
                    
                    # Logic: If we already found a way to reach this time bucket 
                    # with MORE data, ignore this path. Otherwise, update it.
                    if new_idx not in next_states or new_size > next_states[new_idx]:
                        next_states[new_idx] = new_size
        
        # PRUNING (Optimization Step)
        # Even with bucketing, we might have:
        # Bucket 100: Size 500
        # Bucket 101: Size 400 (Slower AND Smaller -> Useless)
        # We must remove these dominated states to keep dictionary small.
        
        # Sort by Time (Index)
        sorted_items = sorted(next_states.items()) # List of (time_idx, size)
        
        pruned_states = {}
        max_size_seen = -1.0
        
        for t_idx, size in sorted_items:
            if size > max_size_seen:
                pruned_states[t_idx] = size
                max_size_seen = size
        
        current_states = pruned_states

    end_bench = time.perf_counter()
    
    if not current_states:
        return 0.0, (end_bench - start_bench)
        
    return max(current_states.values()), (end_bench - start_bench)

def solve_buffer_constrained_schedule_orig(compute_times, N, options, final_deadline):
    """
    Solves the maximization problem with buffer reuse constraints and a final system deadline.
    
    Args:
        compute_times: List of T floats (time to compute each task)
        N: Integer (number of buffers)
        options: List of T lists. options[i] = [(size, duration), ...]
        final_deadline: Float (Time by which the LAST transmission must finish)
        
    Returns:
        (max_size, execution_time_seconds)
    """
    start_bench = time.perf_counter()
    
    T = len(compute_times)
    
    # --- 1. Precompute Timeline Constraints ---
    # Arrival[i]: Time Task i is computed and ready to enter the generic buffer.
    arrivals = []
    current_clock = 0.0
    for t in compute_times:
        current_clock += t
        arrivals.append(current_clock)
        
    # Deadline[i]: Time Task i must be fully transmitted/cleared.
    # Constraint A: Buffer reuse. Task i must clear before Task i+N finishes compute.
    # Constraint B: Global system deadline.
    task_deadlines = []
    for i in range(T):
        constraints = []
        
        # Buffer Constraint (only exists if there is a Task i+N)
        if i + N < T:
            constraints.append(arrivals[i + N])
        else:
            # If no task overwrites this buffer, buffer constraint is infinite
            constraints.append(float('inf'))
            
        # Global Deadline Constraint (applies to everyone, effectively)
        constraints.append(final_deadline)
        
        task_deadlines.append(min(constraints))

    # --- 2. Pareto-Frontier Dynamic Programming ---
    # State format: (finish_time, accumulated_size)
    # We maintain a list of valid states at the completion of task i-1
    current_states = [(0.0, 0.0)] 
    
    for i in range(T):
        next_states = []
        
        arrival = arrivals[i]
        deadline = task_deadlines[i]
        task_opts = options[i]
        
        # If no states survived the previous round, optimization failed.
        if not current_states:
            end_bench = time.perf_counter()
            print(f"DEBUG: Optimization died at Task {i} (No valid paths)")
            return 0.0, end_bench - start_bench

        # Expansion Step: Try all options for all previous valid states
        for prev_finish, prev_size in current_states:
            # We can start sending only when task is computed AND channel is free
            start_send = max(prev_finish, arrival)
            
            for (opt_size, opt_dur) in task_opts:
                finish_send = start_send + opt_dur
                
                # Check strict deadline
                if finish_send <= deadline:
                    next_states.append((finish_send, prev_size + opt_size))
        
        # --- 3. Pruning Step (The Performance Engine) ---
        # Sort by Finish Time.
        # We only keep a state if it offers more Size than all "faster" states.
        
        if not next_states:
            # Dead end
            current_states = []
            continue
            
        next_states.sort(key=lambda x: x[0]) # Sort by finish time ASC
        
        pruned = []
        max_size_seen = -1.0
        
        for f_time, total_sz in next_states:
            # Only keep if this state yields more data than any state that finished earlier
            if total_sz > max_size_seen:
                pruned.append((f_time, total_sz))
                max_size_seen = total_sz
                
        current_states = pruned

    end_bench = time.perf_counter()
    
    if not current_states:
        return 0.0, end_bench - start_bench
        
    # Result is the max size of the surviving states
    return max(s[1] for s in current_states), end_bench - start_bench

# --- TEST CASE GENERATOR ---

def run_benchmark():
    print("--- Generating Test Case ---")
    T = 64
    N = 10
    k = 4
    
    # 1. Generate Compute Times (avg 10.0 ms)
    # Total compute time will be approx 640ms
    compute_times = [random.uniform(8.0, 12.0) for _ in range(T)]
    
    # 2. Generate Options (Size, Duration)
    # We make durations average around 10ms so the network is slightly congested relative to compute
    # To make it hard: High size = High duration
    options = []
    for _ in range(T):
        task_opts = []
        base_dur = random.uniform(5.0, 15.0)
        base_size = base_dur * 10 # Base throughput
        
        # Create k variations
        for _ in range(k):
            # Variance factor
            factor = random.uniform(0.8, 1.2)
            dur = base_dur * factor
            size = base_size * factor * random.uniform(0.9, 1.1) # Add some noise to throughput
            task_opts.append((int(size), dur))
        options.append(task_opts)

    # 3. Set Final Deadline
    # If tasks compute in ~640ms, let's set a deadline at 750ms
    # This forces the system to be efficient but is achievable.
    total_compute = sum(compute_times)
    final_deadline = total_compute * 1.2 
    
    print(f"Parameters: T={T}, N={N}, Options per Task={k}")
    print(f"Total Compute Time: {total_compute:.2f}ms")
    print(f"Final Deadline:     {final_deadline:.2f}ms")
    print("-" * 30)

    # --- RUN SOLVER ---
    print("Running Solver...")
    max_size, duration = solve_buffer_constrained_schedule(compute_times, N, options, final_deadline)
    
    print("-" * 30)
    print(f"OPTIMIZATION COMPLETE")
    print(f"Max Data Sent:      {max_size:.0f} bits")
    print(f"Solver Time:        {duration*1000:.4f} ms") # Convert to milliseconds
    print("-" * 30)
    
    if max_size == 0:
        print("Note: If Max Data is 0, the constraints were too tight (impossible problem).")

if __name__ == "__main__":
    run_benchmark()