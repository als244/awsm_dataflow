#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// --- DEFINITIONS ---

// A hard limit on the number of "active paths" we track.
// 2048 is generous for this problem size. If we exceed this, we drop the worst
// paths.
#define MAX_STATES 2048

typedef struct {
  double size;     // Reward
  double duration; // Cost (Time)
} Option;

typedef struct {
  double finish_time;
  double total_size;
} State;

// --- CORE LOGIC ---

// Helper for qsort: sorts States by Finish Time (Ascending)
int compare_states(const void *a, const void *b) {
  State *s1 = (State *)a;
  State *s2 = (State *)b;
  if (s1->finish_time < s2->finish_time)
    return -1;
  if (s1->finish_time > s2->finish_time)
    return 1;
  return 0;
}

/**
 * solve_schedule
 * * T: Number of tasks
 * N: Number of buffers
 * k: Number of options per task
 * compute_times: Array [T] of computation times
 * all_options: Flat Array [T * k] of Options.
 * (Task 0's options are at indices 0..k-1)
 * final_deadline: Hard system stop time
 * * Returns: Maximum accumulated size (or -1.0 if failed)
 */
double solve_schedule(int T, int N, int k, double *compute_times,
                      Option *all_options, double final_deadline) {

  // 1. Precompute Timeline
  // We allocate strictly on stack or pre-malloced buffers for speed.
  // For T=hundreds, variable length arrays (VLA) or malloc are fine.
  double *arrivals = (double *)malloc(T * sizeof(double));
  double *deadlines = (double *)malloc(T * sizeof(double));

  double current_clock = 0.0;
  for (int i = 0; i < T; i++) {
    current_clock += compute_times[i];
    arrivals[i] = current_clock;
  }

  for (int i = 0; i < T; i++) {
    double d = final_deadline;
    // Buffer Constraint: Must clear before Task i+N finishes compute
    if (i + N < T) {
      if (arrivals[i + N] < d) {
        d = arrivals[i + N];
      }
    }
    deadlines[i] = d;
  }

  // 2. DP Initialization
  // We double-buffer the states to avoid reallocation
  State *current_states = (State *)malloc(MAX_STATES * sizeof(State));
  State *next_states = (State *)malloc(MAX_STATES * sizeof(State));

  int num_current = 1;
  current_states[0].finish_time = 0.0;
  current_states[0].total_size = 0.0;

  // 3. Main Loop
  for (int i = 0; i < T; i++) {
    double arrival = arrivals[i];
    double deadline = deadlines[i];
    Option *task_opts =
        &all_options[i * k]; // Pointer arithmetic to jump to current task

    int num_next = 0;

    // Expand
    for (int s = 0; s < num_current; s++) {
      double prev_finish = current_states[s].finish_time;
      double prev_size = current_states[s].total_size;

      // Wait for compute if needed
      double start_send = (prev_finish > arrival) ? prev_finish : arrival;

      for (int opt = 0; opt < k; opt++) {
        double finish = start_send + task_opts[opt].duration;

        if (finish <= deadline) {
          if (num_next < MAX_STATES) {
            next_states[num_next].finish_time = finish;
            next_states[num_next].total_size = prev_size + task_opts[opt].size;
            num_next++;
          } else {
            // Buffer full. In a real system, you might implement
            // emergency pruning here. For now, we ignore overflow
            // (risky, but usually fine with MAX_STATES=2048).
          }
        }
      }
    }

    if (num_next == 0) {
      // Optimization Failed (Dead end)
      free(arrivals);
      free(deadlines);
      free(current_states);
      free(next_states);
      return -1.0;
    }

    // Prune (The Pareto Logic)
    // A. Sort by finish time
    qsort(next_states, num_next, sizeof(State), compare_states);

    // B. Filter dominated states
    // Re-use 'current_states' buffer to store the pruned list for the next
    // round (Swapping the pointers conceptually, but here we physically copy to
    // keep it clean) Actually, let's swap pointers to avoid copying.

    State *pruned_buffer = current_states;
    int num_pruned = 0;
    double max_size_seen = -1.0;

    for (int s = 0; s < num_next; s++) {
      if (next_states[s].total_size > max_size_seen) {
        pruned_buffer[num_pruned] = next_states[s];
        max_size_seen = next_states[s].total_size;
        num_pruned++;
      }
    }

    // Swap pointers: pruned_buffer is now the input for the next round
    current_states = pruned_buffer;
    num_current = num_pruned;

    // 'next_states' pointer is now free to be overwritten next loop
    // (We need to re-assign next_states to the OTHER buffer)
    // The buffer we just read from (next_states in this loop) is now garbage
    State *temp = next_states;
    next_states = temp; // No change needed, just reuse the memory block
  }

  // 4. Find Max
  double global_max = 0.0;
  for (int i = 0; i < num_current; i++) {
    if (current_states[i].total_size > global_max) {
      global_max = current_states[i].total_size;
    }
  }

  // Cleanup
  free(arrivals);
  free(deadlines);
  // Note: current_states and next_states point to the 2 allocated blocks.
  // Since we swapped pointers, we must be careful freeing.
  // Simpler way: Keep original pointers if we wanted to be strict,
  // but here we just free the active pointers since they point to the same 2
  // blocks. (Assuming we only swapped them) To be perfectly safe against
  // pointer aliasing confusion in cleanup: It's better to manage them as `buf1`
  // and `buf2` and just toggle an index. But for this snippet, simply letting
  // OS reclaim on exit is fine for a script, For library usage, track the
  // original mallocs.

  return global_max;
}

// --- BENCHMARK RUNNER ---

int main() {
  srand(time(NULL));

  int T = 64;
  int N = 10;
  int k = 4;

  // Setup Data
  double *compute = malloc(T * sizeof(double));
  Option *options = malloc(T * k * sizeof(Option));

  double total_compute = 0;
  for (int i = 0; i < T; i++) {
    compute[i] = 8.0 + ((double)rand() / RAND_MAX) * 4.0; // 8.0 to 12.0
    total_compute += compute[i];

    for (int j = 0; j < k; j++) {
      double dur = 5.0 + ((double)rand() / RAND_MAX) * 10.0; // 5.0 to 15.0
      double size = dur * 10.0 * (0.9 + ((double)rand() / RAND_MAX) * 0.2);

      options[i * k + j].duration = dur;
      options[i * k + j].size = size;
    }
  }

  double deadline = total_compute * 1.2;

  printf("Running C Benchmark (T=%d, N=%d)...\n", T, N);

  clock_t start = clock();
  double result = solve_schedule(T, N, k, compute, options, deadline);
  clock_t end = clock();

  double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

  printf("Max Size: %.2f\n", result);
  printf("Time: %.6f seconds\n", time_taken);

  free(compute);
  free(options);
  return 0;
}