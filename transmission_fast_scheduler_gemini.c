#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// --- TUNING ---
#define TIME_SCALE 10
// 16384 ticks = ~1.6 seconds. Sufficient for T=64 (approx 800ms).
// Must be multiple of 64 for bitset alignment.
#define MAX_TICKS 16384
#define WORD_COUNT (MAX_TICKS / 64)

typedef struct {
  int duration_ticks;
  double size;
} FastOption;

// --- SOLVER ---
double solve_bitset(int T, int N, int k, double *compute_times,
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
  FastOption opts[2048];
  for (int i = 0; i < T * k; i++) {
    opts[i].duration_ticks = (int)(raw_durations[i] * TIME_SCALE);
    if (opts[i].duration_ticks < 1)
      opts[i].duration_ticks = 1;
    opts[i].size = raw_sizes[i];
  }

  // 3. Bitset & DP Arrays
  // active_words: Bits set to 1 indicate a valid state exists at that index.
  // dp: Stores the max size for that index.
  static uint64_t current_words[WORD_COUNT];
  static uint64_t next_words[WORD_COUNT];
  static double dp[MAX_TICKS]; // Shared DP array

  // Initialization
  // We clear the bitsets. We DO NOT need to clear 'dp' (we rely on bits).
  memset(current_words, 0, sizeof(current_words));
  memset(next_words, 0, sizeof(next_words));

  // Setup Start State (Time 0)
  current_words[0] = 1ULL; // Set 0th bit
  dp[0] = 0.0;

  // Track bounds to avoid scanning all 256 words
  int max_word_idx = 0;

  // --- MAIN LOOP ---
  for (int i = 0; i < T; i++) {
    int arrival = arrivals[i];
    int deadline = deadlines[i];
    FastOption *task_opts = &opts[i * k];

    int found_any = 0;
    int new_max_word = -1;

    // Iterate only up to the highest active word
    for (int w = 0; w <= max_word_idx; w++) {
      uint64_t word = current_words[w];

      // SUPER FAST SKIP: If 64 ticks are empty, continue instantly.
      if (word == 0)
        continue;

      // Clear this word from current (Prepare for next round reuse)
      current_words[w] = 0;

      // Process all set bits in this word
      while (word != 0) {
        // Find index of least significant bit (0-63)
        int bit_idx = __builtin_ctzll(word);
        int t = (w * 64) + bit_idx;

        // Clear the bit so we don't process it again
        word &= ~(1ULL << bit_idx); // or word ^= (1ULL << bit_idx)

        double current_size = dp[t];

        // Wait Logic
        int start_t = (t > arrival) ? t : arrival;

        for (int opt = 0; opt < k; opt++) {
          int finish = start_t + task_opts[opt].duration_ticks;

          if (finish <= deadline && finish < MAX_TICKS) {
            double new_size = current_size + task_opts[opt].size;

            int fin_word = finish / 64;
            int fin_bit = finish % 64;
            uint64_t mask = (1ULL << fin_bit);

            // Check if we already visited this state in the 'next' generation
            if (next_words[fin_word] & mask) {
              // Collision: Update Max
              if (new_size > dp[finish]) {
                dp[finish] = new_size;
              }
            } else {
              // First visit: Set bit and Value
              next_words[fin_word] |= mask;
              dp[finish] = new_size;
            }

            if (fin_word > new_max_word)
              new_max_word = fin_word;
            found_any = 1;
          }
        }
      }
    }

    if (!found_any)
      return 0.0;

    // Swap Bitsets Logic
    // We actually just swap the pointers logic, but since they are static
    // arrays and we clear 'current' as we go, we can just memcpy 'next' to
    // 'current'? No, memcpy is slow (2KB). Better: Swap pointers. But
    // current_words is static array, not pointer. Let's use pointers. Wait, for
    // static arrays we can't just swap symbols. We will just copy 'next' to
    // 'current' efficiently? Actually, since we track 'new_max_word', we only
    // copy the relevant chunks.

    // Pointer Swap Implementation:
    // (We need to declare pointers outside loop to swap them properly)
    // Refactored below for pointer swapping.
    // For this inline version, let's just do a manual copy loop, it's very fast
    // because we only copy up to new_max_word.

    max_word_idx = new_max_word;
    for (int w = 0; w <= new_max_word; w++) {
      current_words[w] = next_words[w];
      next_words[w] = 0; // Clear next for future usage
    }
  }

  // Final Result Scan
  double global_max = 0.0;
  for (int w = 0; w <= max_word_idx; w++) {
    uint64_t word = current_words[w];
    while (word != 0) {
      int bit_idx = __builtin_ctzll(word);
      word &= ~(1ULL << bit_idx);
      int t = (w * 64) + bit_idx;
      if (dp[t] > global_max)
        global_max = dp[t];
    }
  }

  return global_max;
}

// --- BENCHMARK ---
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
  solve_bitset(T, N, k, compute, durations, sizes, deadline);

  int iterations = 20000;
  clock_t start = clock();

  volatile double result;
  for (int i = 0; i < iterations; i++) {
    result = solve_bitset(T, N, k, compute, durations, sizes, deadline);
  }

  clock_t end = clock();
  double time_per_run = ((double)(end - start)) / CLOCKS_PER_SEC / iterations;

  printf("Result: %.2f\n", result);
  printf("Time per run: %.2f microseconds\n", time_per_run * 1000000);

  return 0;
}