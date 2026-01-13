import time
import math

def solve_fast_quantized(compute_times, N, options, final_deadline, time_resolution=1.0):
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