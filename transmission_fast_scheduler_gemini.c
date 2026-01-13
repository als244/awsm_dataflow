#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// --- TUNING ---
// 10 = 0.1ms precision.
#define TIME_SCALE 10
// Max horizon. 2000ms * 10 = 20000.
#define MAX_TICKS 30000
// Max number of active paths we expect (heuristic constraint to prevent stack
// overflow)
#define MAX_ACTIVE_PATHS 2048

typedef struct {
  int duration_ticks;
  double size;
} FastOption;

// --- SOLVER ---

double solve_ultra(int T, int N, int k, double *compute_times,
                   double *raw_durations, double *raw_sizes,
                   double final_deadline) {

  // 1. Precompute Arrivals & Deadlines
  int arrivals[1024];
  int deadlines[1024];
  int deadline_ticks = (int)(final_deadline * TIME_SCALE);

  double current_clock = 0.0;
  for (int i = 0; i < T; i++) {
    current_clock += compute_times[i];
    arrivals[i] = (int)(current_clock * TIME_SCALE);
  }

  for (int i = 0; i < T; i++) {
    int d = deadline_ticks;
    if (i + N < T) {
      if (arrivals[i + N] < d)
        d = arrivals[i + N];
    }
    deadlines[i] = d;
  }

  // 2. Precompute Options
  FastOption opts[4096]; // Max T=1024, k=4
  for (int i = 0; i < T * k; i++) {
    opts[i].duration_ticks = (int)(raw_durations[i] * TIME_SCALE);
    if (opts[i].duration_ticks < 1)
      opts[i].duration_ticks = 1;
    opts[i].size = raw_sizes[i];
  }

  // 3. The Sparse-Dense Hybrid
  // dp[t] stores the max size at time t. -1.0 means empty.
  static double dp[MAX_TICKS];

  // We use two lists to track WHICH indices in dp[] are valid.
  // This allows us to jump directly to valid data.
  static int current_indices[MAX_ACTIVE_PATHS];
  static int next_indices[MAX_ACTIVE_PATHS];

  int num_current = 0;
  int num_next = 0;

  // Initialization (Clear DP array ONCE)
  // For extreme speed, we don't memset the whole thing every run if we track
  // dirty bits, but memset is fast enough (approx 10us for 200KB).
  memset(dp, 0, sizeof(dp)); // actually 0.0 is valid for size, so we need a
                             // flag? optimization: assume 0.0 size is default
                             // empty. if real size can be 0, we need -1.
  for (int i = 0; i < MAX_TICKS; i++)
    dp[i] = -1.0;

  // Setup Start State
  dp[0] = 0.0;
  current_indices[0] = 0; // The index '0' is valid
  num_current = 1;

  // --- MAIN LOOP ---
  for (int i = 0; i < T; i++) {
    int arrival = arrivals[i];
    int deadline = deadlines[i];
    FastOption *task_opts = &opts[i * k];

    num_next = 0;

    // Iterate ONLY the valid indices from previous step
    for (int idx_ptr = 0; idx_ptr < num_current; idx_ptr++) {
      int t = current_indices[idx_ptr];
      double prev_size = dp[t];

      // Clear the old state from DP array as we process it
      // (Lazy clearing: Prepare dp[] for next round usage)
      // Wait! We can't clear yet because multiple paths might merge to this
      // node? Actually, we are reading from 'current' (time t) and writing to
      // 'next' (time finish). We should clear 't' after we are done with it?
      // Safer strategy: Clear dp[t] AFTER the loop or track what we touched.

      int start_tick = (t > arrival) ? t : arrival;

      for (int opt = 0; opt < k; opt++) {
        int finish = start_tick + task_opts[opt].duration_ticks;

        if (finish <= deadline && finish < MAX_TICKS) {
          double new_size = prev_size + task_opts[opt].size;

          // CHECK: Is this the first time we reached 'finish' this round?
          if (dp[finish] == -1.0) {
            // It's a new valid state for next round
            if (num_next < MAX_ACTIVE_PATHS) {
              next_indices[num_next++] = finish;
              dp[finish] = new_size;
            }
          } else {
            // We already reached this time via another path. Update max.
            if (new_size > dp[finish]) {
              dp[finish] = new_size;
            }
          }
        }
      }

      // Clean up the 'prev' slot we just read from.
      // IMPORTANT: Only if we are sure no future 'next' writes will collide
      // with this 'prev' slot IF 'finish' < 't' (impossible since duration >
      // 0). So it is safe to clear dp[t] now? No, because 'next_indices' might
      // include 't' again (if duration is small and arrival matches). We must
      // clear dp[t] strictly after we are done writing everything? Actually, we
      // need to separate 'Read' (current) and 'Write' (next) phases if we use
      // one array. But we are using one array 'dp'. If we overwrite dp[finish],
      // and finish happens to be a future 't' in current_indices? Since we
      // iterate t in arbitrary order (or sorted), and finish > t always, we are
      // safe IF we iterate t descending? No, finish > t. Safe strategy: We need
      // a temporary buffer or we risk reading partial updates? With positive
      // duration, finish > t. So we write "ahead". If we process t=10, write
      // to 20. Later we process t=20? Yes! Collision risk if current_indices
      // contains 20. Solution: We need 2 DP arrays (ping-pong) OR verify logic.
      // Current indices contains "states reached after task i-1".
      // Next indices contains "states reached after task i".
      // Task i takes time. So finish > start >= t.
      // So we are strictly moving forward in time.
      // But 'dp' array mixes generations.
      // FIX: Use 2 DP arrays (dp_read, dp_write) to be safe and fast.
    }

    // --- 4. PRUNING and SWAP ---

    // Since we didn't implement the 2-array logic above, let's fix the logic
    // here. We will clear the "old" indices now. But we already wrote new
    // values into dp[] at 'finish' indices. If there was overlap (e.g. t=10 was
    // valid, and t=5+dur=5 -> 10 became valid for next), we have a conflict.

    // CORRECTED LOOP for Single Array with Collision Safety:
    // We can't easily do it with 1 array in 1 pass without sorting.
    // Let's use the Ping-Pong DP approach. It's robust and fast.
  }
  return 0.0; // Placeholder, see corrected function below
}

// --- CORRECTED SOLVER FUNCTION ---

double solve_optimized(int T, int N, int k, double *compute_times,
                       double *raw_durations, double *raw_sizes,
                       double final_deadline) {

  int arrivals[1024];
  int deadlines[1024];
  int deadline_ticks = (int)(final_deadline * TIME_SCALE);

  double current_clock = 0.0;
  for (int i = 0; i < T; i++) {
    current_clock += compute_times[i];
    arrivals[i] = (int)(current_clock * TIME_SCALE);
  }

  for (int i = 0; i < T; i++) {
    int d = deadline_ticks;
    if (i + N < T) {
      if (arrivals[i + N] < d)
        d = arrivals[i + N];
    }
    deadlines[i] = d;
  }

  FastOption opts[4096];
  for (int i = 0; i < T * k; i++) {
    opts[i].duration_ticks = (int)(raw_durations[i] * TIME_SCALE);
    if (opts[i].duration_ticks < 1)
      opts[i].duration_ticks = 1;
    opts[i].size = raw_sizes[i];
  }

  // Ping-Pong Buffers
  static double dp_A[MAX_TICKS];
  static double dp_B[MAX_TICKS];
  static int indices_A[MAX_ACTIVE_PATHS];
  static int indices_B[MAX_ACTIVE_PATHS];

  // Initial Clear
  for (int i = 0; i < MAX_TICKS; i++) {
    dp_A[i] = -1.0;
    dp_B[i] = -1.0;
  }

  // Pointers to swap
  double *dp_read = dp_A;
  double *dp_write = dp_B;
  int *idx_read = indices_A;
  int *idx_write = indices_B;

  int cnt_read = 1;
  dp_read[0] = 0.0;
  idx_read[0] = 0;

  for (int i = 0; i < T; i++) {
    int arrival = arrivals[i];
    int deadline = deadlines[i];
    FastOption *task_opts = &opts[i * k];

    int cnt_write = 0;

    // SCAN SPARSE LIST
    for (int r = 0; r < cnt_read; r++) {
      int t = idx_read[r];
      double prev_size = dp_read[t];

      // Clean up read-buffer for next reuse (reset to -1)
      dp_read[t] = -1.0;

      int start_tick = (t > arrival) ? t : arrival;

      for (int opt = 0; opt < k; opt++) {
        int finish = start_tick + task_opts[opt].duration_ticks;

        if (finish <= deadline && finish < MAX_TICKS) {
          double new_size = prev_size + task_opts[opt].size;

          if (dp_write[finish] == -1.0) {
            // First time visiting this time-slot in this round
            if (cnt_write < MAX_ACTIVE_PATHS) {
              idx_write[cnt_write++] = finish;
              dp_write[finish] = new_size;
            }
          } else {
            // Collision: Keep max
            if (new_size > dp_write[finish]) {
              dp_write[finish] = new_size;
            }
          }
        }
      }
    }

    // Swap Pointers
    double *temp_dp = dp_read;
    dp_read = dp_write;
    dp_write = temp_dp;
    int *temp_idx = idx_read;
    idx_read = idx_write;
    idx_write = temp_idx;
    cnt_read = cnt_write;

    if (cnt_read == 0)
      return 0.0; // Fail
  }

  // Final Search
  double global_max = 0.0;
  for (int i = 0; i < cnt_read; i++) {
    int t = idx_read[i];
    if (dp_read[t] > global_max)
      global_max = dp_read[t];
    // Cleanup last buffer (optional, but good practice)
    dp_read[t] = -1.0;
  }

  return global_max;
}

int main() {
  int T = 64;
  int N = 10;
  int k = 4;

  double compute[64];
  double durations[256];
  double sizes[256];

  srand(1234);
  double total_time = 0;
  for (int i = 0; i < T; i++) {
    compute[i] = 8.0 + ((double)rand() / RAND_MAX) * 4.0;
    total_time += compute[i];
  }
  for (int i = 0; i < T * k; i++) {
    durations[i] = 5.0 + ((double)rand() / RAND_MAX) * 10.0;
    sizes[i] = durations[i] * 10.0;
  }
  double deadline = total_time * 1.2;

  // Warmup
  solve_optimized(T, N, k, compute, durations, sizes, deadline);

  int iterations = 10000;
  clock_t start = clock();

  volatile double result;
  for (int i = 0; i < iterations; i++) {
    result = solve_optimized(T, N, k, compute, durations, sizes, deadline);
  }

  clock_t end = clock();
  double time_per_run = ((double)(end - start)) / CLOCKS_PER_SEC / iterations;

  printf("Result: %.2f\n", result);
  printf("Time per run: %.2f microseconds\n", time_per_run * 1000000);

  return 0;
}