#include "transmission_scheduler.h"
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- CONFIGURATION ---
// 1 tick = 1 ms., 10 ticks = 100 us
// Buffer covers ~100 seconds.

// Solver is faster when BUF_SIZE fits in cache (e.g. 64KB), but then 
// to achieve 100us precision we are limited in duration of scheduled tasks.
// This config should take solver be on order of hundreds of microseconds, 
// could also do something like TIME_SCALE = 1 and BUF_SIZE = 65536 for order of 
// magnitude faster solving speed
#define TIME_SCALE 10
#define BUF_SIZE 1048576
#define INF_NEG -1.0e18

typedef struct {
  int duration_ticks;
  double size;
} FastOption;

// =========================================================
//  SCALAR SOLVER (Portable Fallback)
// =========================================================
static double solve_scalar(int T, int N, int k, const double *compute,
                           const double *durations, const double *sizes,
                           double deadline, int *out_choices) {

  // 1. Precompute integer constraints
  int *arrivals = (int *)malloc(T * sizeof(int));
  int *deadlines = (int *)malloc(T * sizeof(int));
  if (!arrivals || !deadlines)
    return 0.0;

  int dead_ticks = (int)(deadline * TIME_SCALE);
  double clk = 0;
  for (int i = 0; i < T; i++) {
    clk += compute[i];
    arrivals[i] = (int)(clk * TIME_SCALE);
  }

  for (int i = 0; i < T; i++) {
    // 1. Start with the Global Deadline
    int d = dead_ticks;

    // 2. The Hard Buffer Constraint
    // Task i transfer must complete by the time Task i+N-1 finishes compute.
    // If it finishes later, there is idle time waiting for prior buffer to be released -> INVALID.
    if (i + N - 1 < T && i + N - 1 >= 0) {
      if (arrivals[i + N - 1] < d) {
        d = arrivals[i + N - 1];
      }
    }

    deadlines[i] = d;
  }

  // 2. Precompute Options
  FastOption *opts = (FastOption *)malloc(T * k * sizeof(FastOption));
  if (!opts) {
    free(arrivals);
    free(deadlines);
    return 0.0;
  }

  for (int i = 0; i < T * k; i++) {
    opts[i].duration_ticks = (int)(durations[i] * TIME_SCALE);
    if (opts[i].duration_ticks < 1)
      opts[i].duration_ticks = 1;
    opts[i].size = sizes[i];
  }

  // 3. Memory Setup
  double *dp_curr = (double *)malloc(BUF_SIZE * sizeof(double));
  double *dp_next = (double *)malloc(BUF_SIZE * sizeof(double));
  int8_t *history = (int8_t *)malloc((size_t)T * BUF_SIZE * sizeof(int8_t));

  if (!dp_curr || !dp_next || !history) {
    free(arrivals);
    free(deadlines);
    free(opts);
    if (dp_curr)
      free(dp_curr);
    if (dp_next)
      free(dp_next);
    if (history)
      free(history);
    return 0.0;
  }

  for (int i = 0; i < BUF_SIZE; i++) {
    dp_curr[i] = INF_NEG;
    dp_next[i] = INF_NEG;
  }
  dp_curr[0] = 0.0;

  int min_t = 0;
  int max_t = 0;

  // 4. Forward Pass
  for (int i = 0; i < T; i++) {
    int arrival = arrivals[i];
    int limit = deadlines[i];
    FastOption *task_opts = &opts[i * k];

    int next_min = BUF_SIZE;
    int next_max = -1;

    // --- SMART CLEAR (Scalar) ---
    // Find max duration to bound the clear
    int max_dur_task = 0;
    for (int o = 0; o < k; o++)
      if (task_opts[o].duration_ticks > max_dur_task)
        max_dur_task = task_opts[o].duration_ticks;

    // We only need to clear [arrival ... max_t + max_dur]
    // But for safety against "Wait Logic" reading old garbage, we stick to
    // clearing what we write. Wait Logic reads from 'dp_curr' (which is valid).
    // Pull Logic writes to 'dp_next'.
    // So we just need to clear the region in dp_next we might touch.
    int clear_start = arrival;
    int clear_end = max_t + max_dur_task + 1;
    if (clear_end >= BUF_SIZE)
      clear_end = BUF_SIZE;
    if (clear_start > clear_end)
      clear_start = clear_end; // Edge case

    for (int t = clear_start; t < clear_end; t++)
      dp_next[t] = INF_NEG;
    // --- END SMART CLEAR ---

    for (int opt = 0; opt < k; opt++) {
      int dur = task_opts[opt].duration_ticks;
      double size = task_opts[opt].size;

      // Wait Logic
      double max_wait = INF_NEG;
      int search_lim = (max_t < arrival) ? max_t : arrival;
      for (int t = min_t; t <= search_lim; t++) {
        if (dp_curr[t] > max_wait)
          max_wait = dp_curr[t];
      }

      if (max_wait > INF_NEG) {
        int finish = arrival + dur;
        if (finish <= limit && finish < BUF_SIZE) {
          double val = max_wait + size;
          if (val > dp_next[finish]) {
            dp_next[finish] = val;
            history[i * BUF_SIZE + finish] = (int8_t)opt;
            if (finish < next_min)
              next_min = finish;
            if (finish > next_max)
              next_max = finish;
          }
        }
      }

      // Pull Logic
      int src_start = arrival + 1;
      if (src_start < min_t)
        src_start = min_t;

      if (src_start <= max_t) {
        for (int src = src_start; src <= max_t; src++) {
          if (dp_curr[src] > INF_NEG) {
            int dest = src + dur;
            if (dest <= limit && dest < BUF_SIZE) {
              double val = dp_curr[src] + size;
              if (val > dp_next[dest]) {
                dp_next[dest] = val;
                history[i * BUF_SIZE + dest] = (int8_t)opt;
                if (dest < next_min)
                  next_min = dest;
                if (dest > next_max)
                  next_max = dest;
              }
            }
          }
        }
      }
    }

    if (next_max == -1) {
      free(arrivals);
      free(deadlines);
      free(opts);
      free(dp_curr);
      free(dp_next);
      free(history);
      return 0.0;
    }

    min_t = next_min;
    max_t = next_max;

    double *tmp = dp_curr;
    dp_curr = dp_next;
    dp_next = tmp;
  }

  // 5. Traceback
  double max_val = INF_NEG;
  int curr_t = -1;
  for (int t = min_t; t <= max_t; t++) {
    if (dp_curr[t] > max_val) {
      max_val = dp_curr[t];
      curr_t = t;
    }
  }

  if (curr_t != -1) {
    for (int i = T - 1; i >= 0; i--) {
      int opt = history[i * BUF_SIZE + curr_t];
      out_choices[i] = opt;

      int dur = opts[i * k + opt].duration_ticks;
      int computed_prev = curr_t - dur;

      if (computed_prev <= arrivals[i]) {
        curr_t = 0;
      } else {
        curr_t = computed_prev;
      }
    }
  } else {
    max_val = 0.0;
  }

  free(arrivals);
  free(deadlines);
  free(opts);
  free(dp_curr);
  free(dp_next);
  free(history);

  return max_val;
}

// =========================================================
//  AVX2 SOLVER (High Performance)
// =========================================================

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

__attribute__((target("avx2"))) static double
solve_avx2_impl(int T, int N, int k, const double *compute,
                const double *durations, const double *sizes, double deadline,
                int *out_choices) {

  int *arrivals = (int *)malloc(T * sizeof(int));
  int *deadlines = (int *)malloc(T * sizeof(int));
  if (!arrivals || !deadlines)
    return 0.0;

  int dead_ticks = (int)(deadline * TIME_SCALE);
  double clk = 0;
  for (int i = 0; i < T; i++) {
    clk += compute[i];
    arrivals[i] = (int)(clk * TIME_SCALE);
  }

  for (int i = 0; i < T; i++) {
    // 1. Start with the Global Deadline
    int d = dead_ticks;

    // 2. The Hard Buffer Constraint
    // Task i transfer must complete by the time Task i+N-1 finishes compute.
    // If it finishes later, there is idle time waiting for prior buffer to be released -> INVALID.
    if (i + N - 1 < T && i + N - 1 >= 0) {
      if (arrivals[i + N - 1] < d) {
        d = arrivals[i + N - 1];
      }
    }

    deadlines[i] = d;
  }

  FastOption *opts = (FastOption *)malloc(T * k * sizeof(FastOption));
  if (!opts) {
    free(arrivals);
    free(deadlines);
    return 0.0;
  }

  for (int i = 0; i < T * k; i++) {
    opts[i].duration_ticks = (int)(durations[i] * TIME_SCALE);
    if (opts[i].duration_ticks < 1)
      opts[i].duration_ticks = 1;
    opts[i].size = sizes[i];
  }

  // Full Table Alloc
  static double *dp_table = NULL;
  static int dp_rows = 0;
  static int dp_cols = 0;

  // Check for resize needed
  if (!dp_table || (T + 1) > dp_rows || BUF_SIZE > dp_cols) {
    if (dp_table)
      free(dp_table);
    dp_rows = (T + 1);
    dp_cols = BUF_SIZE;
    dp_table = (double *)malloc((size_t)dp_rows * dp_cols * sizeof(double));
  }
  if (!dp_table) {
    free(arrivals);
    free(deadlines);
    free(opts);
    return 0.0;
  }

  // Init Row 0
  // We only clear row 0 fully once.
  for (int t = 0; t < BUF_SIZE; t++)
    dp_table[t] = INF_NEG;
  dp_table[0] = 0.0;

  int min_t = 0;
  int max_t = 0;

  for (int i = 0; i < T; i++) {
    int arrival = arrivals[i];
    int limit = deadlines[i];
    FastOption *task_opts = &opts[i * k];

    double *src = &dp_table[i * BUF_SIZE];
    double *dst = &dp_table[(i + 1) * BUF_SIZE];

    // --- SMART CLEAR (AVX) ---
    int max_dur_task = 0;
    for (int o = 0; o < k; o++)
      if (task_opts[o].duration_ticks > max_dur_task)
        max_dur_task = task_opts[o].duration_ticks;

    int clear_start = arrival;
    int clear_end = max_t + max_dur_task + 32; // padding for AVX writes
    if (clear_end >= BUF_SIZE)
      clear_end = BUF_SIZE;
    if (clear_start > clear_end)
      clear_start = clear_end;

    // Only clear active region
    for (int t = clear_start; t < clear_end; t++)
      dst[t] = INF_NEG;
    // --- END SMART CLEAR ---

    // Wait Value
    double max_wait = INF_NEG;
    int search_lim = (max_t < arrival) ? max_t : arrival;
    for (int t = min_t; t <= search_lim; t++) {
      if (src[t] > max_wait)
        max_wait = src[t];
    }

    int next_min = BUF_SIZE;
    int next_max = -1;

    if (max_wait > INF_NEG) {
      for (int opt = 0; opt < k; opt++) {
        int fin = arrival + task_opts[opt].duration_ticks;
        if (fin <= limit && fin < BUF_SIZE) {
          double val = max_wait + task_opts[opt].size;
          if (val > dst[fin])
            dst[fin] = val;
          if (fin < next_min)
            next_min = fin;
          if (fin > next_max)
            next_max = fin;
        }
      }
    }

    // AVX Pull
    for (int opt = 0; opt < k; opt++) {
      int dur = task_opts[opt].duration_ticks;
      double size = task_opts[opt].size;
      int start_dst = arrival + 1 + dur;
      int end_dst = max_t + dur;

      if (start_dst > limit)
        continue;
      if (end_dst > limit)
        end_dst = limit;
      if (end_dst >= BUF_SIZE)
        end_dst = BUF_SIZE - 1; // Safety

      if (start_dst < next_min)
        next_min = start_dst;
      if (end_dst > next_max)
        next_max = end_dst;

      __m256d v_size = _mm256_set1_pd(size);

      int t = start_dst;
      for (; t <= end_dst - 3; t += 4) {
        __m256d v_src = _mm256_loadu_pd(&src[t - dur]);
        __m256d v_res = _mm256_add_pd(v_src, v_size);
        __m256d v_dst = _mm256_loadu_pd(&dst[t]);
        v_dst = _mm256_max_pd(v_dst, v_res);
        _mm256_storeu_pd(&dst[t], v_dst);
      }
      for (; t <= end_dst; t++) {
        double val = src[t - dur] + size;
        if (val > dst[t])
          dst[t] = val;
      }
    }

    if (next_max == -1) {
      free(arrivals);
      free(deadlines);
      free(opts);
      // dp_table persists
      return 0.0;
    }
    min_t = next_min;
    max_t = next_max;
  }

  double *final_row = &dp_table[T * BUF_SIZE];
  double max_val = INF_NEG;
  int curr_t = -1;

  // 1. Find the best finish time for the last task
  for (int t = min_t; t <= max_t; t++) {
    if (final_row[t] > max_val) {
      max_val = final_row[t];
      curr_t = t;
    }
  }

  if (curr_t != -1) {
    // 2. Walk backwards
    for (int i = T - 1; i >= 0; i--) {
      double current_val = dp_table[(i + 1) * BUF_SIZE + curr_t];
      double *src_row = &dp_table[i * BUF_SIZE];
      int arrival = arrivals[i];
      FastOption *task_opts = &opts[i * k];

      int best_opt = 0; // Default to 0 to prevent -1
      int best_prev_t = curr_t - task_opts[0].duration_ticks; // Default prev
      double min_error = 1.0e15; // Start with huge error

      // Check all k options to find which one matches 'current_val' best
      for (int opt = 0; opt < k; opt++) {
        int dur = task_opts[opt].duration_ticks;
        double size = task_opts[opt].size;
        int prev_t = curr_t - dur;

        double estimated_val = INF_NEG;
        int candidate_prev_t = -1;

        // A. Pull Logic (Transition from specific prev time)
        if (prev_t > arrival) {
          estimated_val = src_row[prev_t] + size;
          candidate_prev_t = prev_t;
        }
        // B. Wait Logic (Transition from any time <= arrival)
        else if (prev_t <= arrival) {
          // Find max value in valid wait window
          double w_val = INF_NEG;
          int best_z = 0;

          // Optimization: Scan backwards from arrival as later times are
          // preferred
          for (int z = arrival; z >= 0; z--) {
            if (src_row[z] > w_val) {
              w_val = src_row[z];
              best_z = z;
            }
          }
          estimated_val = w_val + size;
          candidate_prev_t = best_z;
        }

        // Calculate Reconstruction Error
        double diff = current_val - estimated_val;
        if (diff < 0)
          diff = -diff; // fabs

        // Keep the option that explains the score with least mathematical error
        if (diff < min_error) {
          min_error = diff;
          best_opt = opt;
          best_prev_t = candidate_prev_t;
        }
      }

      // Assign the winner
      out_choices[i] = best_opt;

      // Setup next iteration
      // Safety: If best_prev_t is somehow invalid (rare), clamp it
      if (best_prev_t < 0)
        best_prev_t = 0;
      curr_t = best_prev_t;
    }
  } else {
    max_val = 0.0;
    // Fill with 0s on total failure
    for (int i = 0; i < T; i++)
      out_choices[i] = 0;
  }

  free(arrivals);
  free(deadlines);
  free(opts);
  return max_val;
}
#endif

// =========================================================
//  PUBLIC DISPATCHER
// =========================================================
double solve_scheduler(int T, int N, int k, const double *compute,
                       const double *durations, const double *sizes,
                       double deadline, int *out_choices) {

#if defined(__x86_64__) || defined(_M_X64)
  if (__builtin_cpu_supports("avx2")) {
    return solve_avx2_impl(T, N, k, compute, durations, sizes, deadline,
                           out_choices);
  }
#endif

  return solve_scalar(T, N, k, compute, durations, sizes, deadline,
                      out_choices);
}