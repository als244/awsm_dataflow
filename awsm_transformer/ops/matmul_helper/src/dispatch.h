#ifndef MATMUL_DISPATCH_H
#define MATMUL_DISPATCH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer to the internal context
typedef struct DispatchContext DispatchContext;

// Create/Destroy
DispatchContext* create_dispatcher(int round_multiple, int max_algos);
void destroy_dispatcher(DispatchContext* ctx);

// Stats
void get_stats(DispatchContext* ctx, uint64_t* out_stats);

// Execution
// Now accepts lda, ldb, ldc, ldd explicitly
int dispatch_matmul(
    DispatchContext* ctx, 
    intptr_t stream_ptr,
    int M, int N, int K,
    intptr_t ptr_A, int lda,
    intptr_t ptr_B, int ldb,
    intptr_t ptr_C, int ldc,
    intptr_t ptr_D, int ldd,
    intptr_t ptr_workspace, size_t workspace_bytes,
    int a_dt, int b_dt, int c_dt, int d_dt, int compute_dt,
    float alpha, float beta,
    int trans_a, int trans_b
);

#ifdef __cplusplus
}
#endif

#endif // MATMUL_DISPATCH_H