#ifndef TRANSMISSION_SCHEDULER_H
#define TRANSMISSION_SCHEDULER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Solve the buffer-constrained maximization problem.
 *
 * @param T             Number of tasks
 * @param N             Number of buffers
 * @param k             Number of options per task
 * @param compute       Array of T compute times
 * @param durations     Array of T*k durations (flattened)
 * @param sizes         Array of T*k sizes (flattened)
 * @param deadline      Final system deadline
 * @param out_choices   Output array of T integers (caller allocated)
 * * @return              The maximum total size, or 0.0 if failed.
 */
double solve_scheduler(int T, int N, int k, const double *compute,
                       const double *durations, const double *sizes,
                       double deadline, int *out_choices);

#ifdef __cplusplus
}
#endif

#endif // TRANSMISSION_SCHEDULER_H