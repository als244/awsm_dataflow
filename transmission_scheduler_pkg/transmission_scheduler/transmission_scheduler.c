#include "transmission_scheduler.h"
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INF_NEG -1.0e18

typedef struct {
  int duration_ticks;
  double size;
} FastOption;

// Per-cell traceback info: which option was chosen and what absolute source
// time produced it.
typedef struct {
  int8_t opt;
  int src_abs;
} TraceCell;

// =========================================================
//  SCALAR SOLVER (Portable Fallback)
// =========================================================
static double solve_scalar(int T, int N, int k, const double *compute,
                           const double *durations, const double *sizes,
                           double deadline, int time_scale, int buf_size,
                           int *out_choices) {

  int *arrivals = (int *)malloc(T * sizeof(int));
  int *deadlines = (int *)malloc(T * sizeof(int));
  if (!arrivals || !deadlines)
    return 0.0;

  int dead_ticks = (int)(deadline * time_scale);
  double clk = 0;
  for (int i = 0; i < T; i++) {
    clk += compute[i];
    arrivals[i] = (int)(clk * time_scale);
  }

  for (int i = 0; i < T; i++) {
    int d = dead_ticks;
    if (i + N - 1 < T && i + N - 1 >= 0) {
      if (arrivals[i + N - 1] < d)
        d = arrivals[i + N - 1];
    }
    deadlines[i] = d;
  }

  FastOption *opts = (FastOption *)malloc(T * k * sizeof(FastOption));
  if (!opts) { free(arrivals); free(deadlines); return 0.0; }

  for (int i = 0; i < T * k; i++) {
    opts[i].duration_ticks = (int)(durations[i] * time_scale);
    if (opts[i].duration_ticks < 1)
      opts[i].duration_ticks = 1;
    opts[i].size = sizes[i];
  }

  double *dp_curr = (double *)malloc(buf_size * sizeof(double));
  double *dp_next = (double *)malloc(buf_size * sizeof(double));
  TraceCell *history = (TraceCell *)malloc((size_t)T * buf_size * sizeof(TraceCell));
  int *win_offsets = (int *)malloc(T * sizeof(int));

  if (!dp_curr || !dp_next || !history || !win_offsets) {
    free(arrivals); free(deadlines); free(opts);
    if (dp_curr) free(dp_curr);
    if (dp_next) free(dp_next);
    if (history) free(history);
    if (win_offsets) free(win_offsets);
    return 0.0;
  }

  for (int i = 0; i < buf_size; i++) {
    dp_curr[i] = INF_NEG;
    dp_next[i] = INF_NEG;
  }

  int base_offset = 0;
  dp_curr[0] = 0.0;
  int min_t = 0, max_t = 0;

  for (int i = 0; i < T; i++) {
    int arrival_abs = arrivals[i];
    int limit_abs = deadlines[i];
    FastOption *task_opts = &opts[i * k];

    int arrival = arrival_abs - base_offset;

    if (arrival >= buf_size / 2) {
      int shift = min_t;
      if (shift > 0) {
        int span = max_t - min_t + 1;
        memmove(&dp_curr[0], &dp_curr[min_t], span * sizeof(double));
        for (int t = span; t < span + shift && t < buf_size; t++)
          dp_curr[t] = INF_NEG;
        base_offset += shift;
        min_t -= shift;
        max_t -= shift;
        arrival -= shift;
      }
    }

    win_offsets[i] = base_offset;
    int next_min = buf_size, next_max = -1;

    int max_dur_task = 0;
    for (int o = 0; o < k; o++)
      if (task_opts[o].duration_ticks > max_dur_task)
        max_dur_task = task_opts[o].duration_ticks;

    int clear_start = (arrival < 0) ? 0 : arrival;
    int clear_end = max_t + max_dur_task + 1;
    if (clear_end > buf_size) clear_end = buf_size;
    if (clear_start > clear_end) clear_start = clear_end;
    for (int t = clear_start; t < clear_end; t++)
      dp_next[t] = INF_NEG;

    double max_wait = INF_NEG;
    int max_wait_abs = 0;
    {
      int search_lim = (max_t < arrival) ? max_t : arrival;
      for (int t = min_t; t <= search_lim; t++) {
        if (dp_curr[t] > max_wait) {
          max_wait = dp_curr[t];
          max_wait_abs = t + base_offset;
        }
      }
    }

    for (int opt = 0; opt < k; opt++) {
      int dur = task_opts[opt].duration_ticks;
      double size = task_opts[opt].size;

      if (max_wait > INF_NEG) {
        int finish = arrival + dur;
        if (finish + base_offset <= limit_abs && finish >= 0 && finish < buf_size) {
          double val = max_wait + size;
          if (val > dp_next[finish]) {
            dp_next[finish] = val;
            TraceCell *cell = &history[(size_t)i * buf_size + finish];
            cell->opt = (int8_t)opt;
            cell->src_abs = max_wait_abs;
            if (finish < next_min) next_min = finish;
            if (finish > next_max) next_max = finish;
          }
        }
      }

      int src_start = arrival + 1;
      if (src_start < min_t) src_start = min_t;

      if (src_start <= max_t) {
        for (int src = src_start; src <= max_t; src++) {
          if (dp_curr[src] > INF_NEG) {
            int dest = src + dur;
            if (dest + base_offset <= limit_abs && dest >= 0 && dest < buf_size) {
              double val = dp_curr[src] + size;
              if (val > dp_next[dest]) {
                dp_next[dest] = val;
                TraceCell *cell = &history[(size_t)i * buf_size + dest];
                cell->opt = (int8_t)opt;
                cell->src_abs = src + base_offset;
                if (dest < next_min) next_min = dest;
                if (dest > next_max) next_max = dest;
              }
            }
          }
        }
      }
    }

    if (next_max == -1) {
      free(arrivals); free(deadlines); free(opts);
      free(dp_curr); free(dp_next); free(history); free(win_offsets);
      return 0.0;
    }

    min_t = next_min;
    max_t = next_max;
    double *tmp = dp_curr; dp_curr = dp_next; dp_next = tmp;
  }

  double max_val = INF_NEG;
  int curr_t = -1;
  for (int t = min_t; t <= max_t; t++) {
    if (dp_curr[t] > max_val) {
      max_val = dp_curr[t];
      curr_t = t;
    }
  }

  if (curr_t != -1) {
    int curr_abs = curr_t + base_offset;
    for (int i = T - 1; i >= 0; i--) {
      int rel_t = curr_abs - win_offsets[i];
      TraceCell *cell = &history[(size_t)i * buf_size + rel_t];
      out_choices[i] = cell->opt;
      curr_abs = cell->src_abs;
    }
  } else {
    max_val = 0.0;
  }

  free(arrivals); free(deadlines); free(opts);
  free(dp_curr); free(dp_next); free(history); free(win_offsets);
  return max_val;
}

// =========================================================
//  AVX2 SOLVER (Vectorized pull with mask-based trace)
// =========================================================

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

__attribute__((target("avx2"))) static double
solve_avx2(int T, int N, int k, const double *compute,
           const double *durations, const double *sizes, double deadline,
           int time_scale, int buf_size, int *out_choices) {

  int *arrivals = (int *)malloc(T * sizeof(int));
  int *deadlines = (int *)malloc(T * sizeof(int));
  if (!arrivals || !deadlines)
    return 0.0;

  int dead_ticks = (int)(deadline * time_scale);
  double clk = 0;
  for (int i = 0; i < T; i++) {
    clk += compute[i];
    arrivals[i] = (int)(clk * time_scale);
  }

  for (int i = 0; i < T; i++) {
    int d = dead_ticks;
    if (i + N - 1 < T && i + N - 1 >= 0) {
      if (arrivals[i + N - 1] < d)
        d = arrivals[i + N - 1];
    }
    deadlines[i] = d;
  }

  FastOption *opts = (FastOption *)malloc(T * k * sizeof(FastOption));
  if (!opts) { free(arrivals); free(deadlines); return 0.0; }

  for (int i = 0; i < T * k; i++) {
    opts[i].duration_ticks = (int)(durations[i] * time_scale);
    if (opts[i].duration_ticks < 1)
      opts[i].duration_ticks = 1;
    opts[i].size = sizes[i];
  }

  double *dp_curr = (double *)malloc(buf_size * sizeof(double));
  double *dp_next = (double *)malloc(buf_size * sizeof(double));
  TraceCell *history = (TraceCell *)malloc((size_t)T * buf_size * sizeof(TraceCell));
  int *win_offsets = (int *)malloc(T * sizeof(int));

  if (!dp_curr || !dp_next || !history || !win_offsets) {
    free(arrivals); free(deadlines); free(opts);
    if (dp_curr) free(dp_curr);
    if (dp_next) free(dp_next);
    if (history) free(history);
    if (win_offsets) free(win_offsets);
    return 0.0;
  }

  for (int i = 0; i < buf_size; i++) {
    dp_curr[i] = INF_NEG;
    dp_next[i] = INF_NEG;
  }

  int base_offset = 0;
  dp_curr[0] = 0.0;
  int min_t = 0, max_t = 0;

  for (int i = 0; i < T; i++) {
    int arrival_abs = arrivals[i];
    int limit_abs = deadlines[i];
    FastOption *task_opts = &opts[i * k];

    int arrival = arrival_abs - base_offset;

    if (arrival >= buf_size / 2) {
      int shift = min_t;
      if (shift > 0) {
        int span = max_t - min_t + 1;
        memmove(&dp_curr[0], &dp_curr[min_t], span * sizeof(double));
        for (int t = span; t < span + shift && t < buf_size; t++)
          dp_curr[t] = INF_NEG;
        base_offset += shift;
        min_t -= shift;
        max_t -= shift;
        arrival -= shift;
      }
    }

    win_offsets[i] = base_offset;
    int next_min = buf_size, next_max = -1;

    int max_dur_task = 0;
    for (int o = 0; o < k; o++)
      if (task_opts[o].duration_ticks > max_dur_task)
        max_dur_task = task_opts[o].duration_ticks;

    int clear_start = (arrival < 0) ? 0 : arrival;
    int clear_end = max_t + max_dur_task + 1;
    if (clear_end > buf_size) clear_end = buf_size;
    if (clear_start > clear_end) clear_start = clear_end;
    for (int t = clear_start; t < clear_end; t++)
      dp_next[t] = INF_NEG;

    double max_wait = INF_NEG;
    int max_wait_abs = 0;
    {
      int search_lim = (max_t < arrival) ? max_t : arrival;
      for (int t = min_t; t <= search_lim; t++) {
        if (dp_curr[t] > max_wait) {
          max_wait = dp_curr[t];
          max_wait_abs = t + base_offset;
        }
      }
    }

    for (int opt = 0; opt < k; opt++) {
      int dur = task_opts[opt].duration_ticks;
      double size = task_opts[opt].size;

      // Wait Logic (scalar — single cell)
      if (max_wait > INF_NEG) {
        int finish = arrival + dur;
        if (finish + base_offset <= limit_abs && finish >= 0 && finish < buf_size) {
          double val = max_wait + size;
          if (val > dp_next[finish]) {
            dp_next[finish] = val;
            TraceCell *cell = &history[(size_t)i * buf_size + finish];
            cell->opt = (int8_t)opt;
            cell->src_abs = max_wait_abs;
            if (finish < next_min) next_min = finish;
            if (finish > next_max) next_max = finish;
          }
        }
      }

      // Pull Logic — AVX2 with mask-based trace recording
      int src_start = arrival + 1;
      if (src_start < min_t) src_start = min_t;

      // Clamp src range so dest stays in bounds
      int src_end = max_t;
      {
        int max_src_dl = limit_abs - base_offset - dur;
        if (max_src_dl < src_end) src_end = max_src_dl;
        int max_src_buf = buf_size - 1 - dur;
        if (max_src_buf < src_end) src_end = max_src_buf;
      }

      if (src_start <= src_end) {
        int dest_start = src_start + dur;
        int dest_end = src_end + dur;
        if (dest_start < next_min) next_min = dest_start;
        if (dest_end > next_max) next_max = dest_end;

        __m256d v_size = _mm256_set1_pd(size);
        TraceCell *hist_row = &history[(size_t)i * buf_size];

        int src = src_start;
        for (; src <= src_end - 3; src += 4) {
          int dest = src + dur;
          __m256d v_src = _mm256_loadu_pd(&dp_curr[src]);
          __m256d v_val = _mm256_add_pd(v_src, v_size);
          __m256d v_old = _mm256_loadu_pd(&dp_next[dest]);
          __m256d v_new = _mm256_max_pd(v_old, v_val);
          _mm256_storeu_pd(&dp_next[dest], v_new);

          // Compare: which lanes did v_val win over v_old?
          // v_val > v_old iff v_new != v_old (since max picked v_val)
          __m256d v_cmp = _mm256_cmp_pd(v_val, v_old, _CMP_GT_OQ);
          int mask = _mm256_movemask_pd(v_cmp);

          // Write trace only for lanes that improved
          if (mask) {
            if (mask & 1) { hist_row[dest].opt = (int8_t)opt; hist_row[dest].src_abs = src + base_offset; }
            if (mask & 2) { hist_row[dest+1].opt = (int8_t)opt; hist_row[dest+1].src_abs = src + 1 + base_offset; }
            if (mask & 4) { hist_row[dest+2].opt = (int8_t)opt; hist_row[dest+2].src_abs = src + 2 + base_offset; }
            if (mask & 8) { hist_row[dest+3].opt = (int8_t)opt; hist_row[dest+3].src_abs = src + 3 + base_offset; }
          }
        }
        // Scalar tail
        for (; src <= src_end; src++) {
          int dest = src + dur;
          double val = dp_curr[src] + size;
          if (val > dp_next[dest]) {
            dp_next[dest] = val;
            hist_row[dest].opt = (int8_t)opt;
            hist_row[dest].src_abs = src + base_offset;
          }
        }
      }
    }

    if (next_max == -1) {
      free(arrivals); free(deadlines); free(opts);
      free(dp_curr); free(dp_next); free(history); free(win_offsets);
      return 0.0;
    }

    min_t = next_min;
    max_t = next_max;
    double *tmp = dp_curr; dp_curr = dp_next; dp_next = tmp;
  }

  double max_val = INF_NEG;
  int curr_t = -1;
  for (int t = min_t; t <= max_t; t++) {
    if (dp_curr[t] > max_val) {
      max_val = dp_curr[t];
      curr_t = t;
    }
  }

  if (curr_t != -1) {
    int curr_abs = curr_t + base_offset;
    for (int i = T - 1; i >= 0; i--) {
      int rel_t = curr_abs - win_offsets[i];
      TraceCell *cell = &history[(size_t)i * buf_size + rel_t];
      out_choices[i] = cell->opt;
      curr_abs = cell->src_abs;
    }
  } else {
    max_val = 0.0;
  }

  free(arrivals); free(deadlines); free(opts);
  free(dp_curr); free(dp_next); free(history); free(win_offsets);
  return max_val;
}
#endif

// =========================================================
//  PUBLIC DISPATCHER
// =========================================================
double solve_scheduler(int T, int N, int k, const double *compute,
                       const double *durations, const double *sizes,
                       double deadline, int time_scale, int buf_size,
                       int *out_choices) {

#if defined(__x86_64__) || defined(_M_X64)
  if (__builtin_cpu_supports("avx2")) {
    return solve_avx2(T, N, k, compute, durations, sizes, deadline,
                      time_scale, buf_size, out_choices);
  }
#endif

  return solve_scalar(T, N, k, compute, durations, sizes, deadline,
                      time_scale, buf_size, out_choices);
}

// =========================================================
//  COMPUTE BUF_SIZE
// =========================================================
int compute_buf_size(int T, int N, int k, const double *compute,
                     const double *durations, double deadline,
                     int time_scale) {

  int *arrivals = (int *)malloc(T * sizeof(int));
  if (!arrivals)
    return 2048;

  double clk = 0;
  for (int i = 0; i < T; i++) {
    clk += compute[i];
    arrivals[i] = (int)(clk * time_scale);
  }

  int dead_ticks = (int)(deadline * time_scale);

  int max_dur = 1, min_dur = 0x7FFFFFFF;
  for (int i = 0; i < T * k; i++) {
    int d = (int)(durations[i] * time_scale);
    if (d < 1) d = 1;
    if (d > max_dur) max_dur = d;
    if (d < min_dur) min_dur = d;
  }

  // Simulate the DP band evolution. Track both sim_min and sim_max
  // (the absolute time range of active states). The buffer must span
  // from sim_min to max(sim_max + max_dur, arrival + max_dur) after
  // each step, because re-centering shifts sim_min to index 0.
  int sim_min = 0;
  int sim_max = 0;
  int max_span = 0;
  for (int i = 0; i < T; i++) {
    int arr = arrivals[i];
    int dl = dead_ticks;
    if (i + N - 1 < T)
      dl = arrivals[i + N - 1];

    // After re-centering, sim_min maps to index 0.
    // Wait logic writes at: (arr - sim_min) + dur  (up to max_dur)
    // Pull logic writes at: (sim_max - sim_min) + dur (up to max_dur)
    // The furthest index written is the max of these two.
    int wait_reach = arr - sim_min + max_dur;
    int pull_reach = sim_max - sim_min + max_dur;
    int reach = (wait_reach > pull_reach) ? wait_reach : pull_reach;
    if (reach > max_span)
      max_span = reach;

    // Update sim_min/sim_max for next step
    int new_min = arr + min_dur;
    int new_max;
    if (sim_max > arr)
      new_max = sim_max + max_dur;
    else
      new_max = arr + max_dur;
    if (new_max > dl)
      new_max = dl;

    sim_min = new_min;
    sim_max = new_max;
  }

  free(arrivals);

  int needed = max_span + 256;
  int buf_size = 1;
  while (buf_size < needed)
    buf_size <<= 1;
  if (buf_size < 2048)
    buf_size = 2048;

  return buf_size;
}
