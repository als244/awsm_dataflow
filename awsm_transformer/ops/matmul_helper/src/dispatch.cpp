#include "dispatch.h"

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <functional>
#include <chrono>
#include <iostream>
#include <cstdio> // Added for setbuf if needed

// ... [Keep Hashing Helpers / Structs exactly as they were] ...

// --- Hashing Helpers ---
inline void hash_combine(std::size_t& seed, int v) {
    seed ^= std::hash<int>{}(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

enum DispatchType {
    DTYPE_FP32 = 0, DTYPE_FP16 = 1, DTYPE_BF16 = 2
};

struct ExactKey {
    int M, N, K;
    int lda, ldb, ldc, ldd; 
    DispatchType a_dt, b_dt, c_dt, d_dt, compute_dt;
    int trans_a, trans_b;

    bool operator==(const ExactKey& o) const {
        return M==o.M && N==o.N && K==o.K &&
               lda==o.lda && ldb==o.ldb && ldc==o.ldc && ldd==o.ldd &&
               a_dt==o.a_dt && b_dt==o.b_dt && c_dt==o.c_dt && d_dt==o.d_dt &&
               compute_dt==o.compute_dt && trans_a==o.trans_a && trans_b==o.trans_b;
    }
};

struct ExactKeyHasher {
    size_t operator()(const ExactKey& k) const {
        size_t h = 0;
        hash_combine(h, k.M); hash_combine(h, k.N); hash_combine(h, k.K);
        hash_combine(h, k.lda); hash_combine(h, k.ldb); 
        hash_combine(h, (int)k.a_dt); 
        hash_combine(h, k.trans_a);
        return h;
    }
};

struct FuzzyKey {
    int M_round, N_round, K_round;
    DispatchType a_dt, b_dt, c_dt, d_dt, compute_dt;
    int trans_a, trans_b;

    bool operator==(const FuzzyKey& o) const {
        return M_round==o.M_round && N_round==o.N_round && K_round==o.K_round &&
               a_dt==o.a_dt && trans_a==o.trans_a && trans_b==o.trans_b;
    }
};

struct FuzzyKeyHasher {
    size_t operator()(const FuzzyKey& k) const {
        size_t h = 0;
        hash_combine(h, k.M_round); hash_combine(h, k.N_round); hash_combine(h, k.K_round);
        hash_combine(h, (int)k.a_dt);
        return h;
    }
};

struct ExactCacheEntry {
    cublasLtMatmulAlgo_t algo;
    cublasLtMatmulDesc_t opDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
};

// --- Context ---
struct DispatchContext {
    cublasLtHandle_t ltHandle;
    std::unordered_map<ExactKey, ExactCacheEntry, ExactKeyHasher> exact_cache;
    std::unordered_map<FuzzyKey, cublasLtMatmulAlgo_t, FuzzyKeyHasher> fuzzy_cache;
    int round_multiple;
    int max_algos;
    
    // Stats
    uint64_t num_algos_saved = 0;
    uint64_t num_matmuls_called = 0;
    uint64_t num_algo_hits = 0;
    
    uint64_t total_cpp_duration_ns = 0;
    uint64_t total_hash_ns = 0;
    uint64_t total_lookup_ns = 0;
    uint64_t total_driver_ns = 0;
};

// --- Helpers ---
static cudaDataType_t get_cuda_dtype(DispatchType dt) {
    if (dt == DTYPE_FP16) return CUDA_R_16F;
    if (dt == DTYPE_BF16) return CUDA_R_16BF;
    return CUDA_R_32F;
}

static cublasComputeType_t get_compute_type(DispatchType dt) {
    return CUBLAS_COMPUTE_32F; 
}

static int round_dim(int dim, int multiple) {
    if (multiple <= 1) return dim;
    if (dim < multiple) return dim; 
    return ((dim + multiple - 1) / multiple) * multiple;
}


// --- API Implementation ---

DispatchContext* create_dispatcher(int round_multiple, int max_algos) {
    // 1. Force stderr to be unbuffered to ensure prints show up immediately in Python
    std::cerr.setf(std::ios::unitbuf);
    
    DispatchContext* ctx = new DispatchContext();
    cublasLtCreate(&ctx->ltHandle);
    ctx->round_multiple = round_multiple;
    ctx->max_algos = (max_algos > 0) ? max_algos : 1;
    return ctx;
}

void destroy_dispatcher(DispatchContext* ctx) {
    if (ctx) {
        for (auto& kv : ctx->exact_cache) {
            cublasLtMatmulDescDestroy(kv.second.opDesc);
            cublasLtMatrixLayoutDestroy(kv.second.Adesc);
            cublasLtMatrixLayoutDestroy(kv.second.Bdesc);
            cublasLtMatrixLayoutDestroy(kv.second.Cdesc);
            cublasLtMatrixLayoutDestroy(kv.second.Ddesc);
        }
        cublasLtDestroy(ctx->ltHandle);
        delete ctx;
    }
}

void get_stats(DispatchContext* ctx, uint64_t* out_stats) {
    if (!ctx) return;
    out_stats[0] = ctx->num_algos_saved;
    out_stats[1] = ctx->num_matmuls_called;
    out_stats[2] = ctx->num_algo_hits;
    out_stats[3] = ctx->total_cpp_duration_ns;
    out_stats[4] = ctx->total_hash_ns;
    out_stats[5] = ctx->total_lookup_ns;
    out_stats[6] = ctx->total_driver_ns;
}

int dispatch_matmul(DispatchContext* ctx, 
                     intptr_t stream_ptr,
                     int M, int N, int K,
                     intptr_t ptr_A, int lda,
                     intptr_t ptr_B, int ldb,
                     intptr_t ptr_C, int ldc,
                     intptr_t ptr_D, int ldd,
                     intptr_t ptr_workspace, size_t workspace_bytes,
                     int a_dt, int b_dt, int c_dt, int d_dt, int compute_dt,
                     float alpha, float beta,
                     int trans_a, int trans_b) 
{
    if (M == 0 || N == 0 || K == 0) return 0; 

    cublasStatus_t status;
    auto start_total = std::chrono::high_resolution_clock::now();
    ctx->num_matmuls_called++;

    cudaStream_t stream = (cudaStream_t)stream_ptr;
    void* d_A = (void*)ptr_A;
    void* d_B = (void*)ptr_B;
    void* d_C = (void*)ptr_C;
    void* d_D = (void*)ptr_D;
    void* d_workspace = (void*)ptr_workspace;

    // --- 1. Hash ---
    ExactKey e_key = {M, N, K, lda, ldb, ldc, ldd, (DispatchType)a_dt, (DispatchType)b_dt, (DispatchType)c_dt, (DispatchType)d_dt, (DispatchType)compute_dt, trans_a, trans_b};
    
    auto t_hash_0 = std::chrono::high_resolution_clock::now();
    volatile size_t h = ExactKeyHasher()(e_key);
    auto t_hash_1 = std::chrono::high_resolution_clock::now();
    ctx->total_hash_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t_hash_1 - t_hash_0).count();

    // --- 2. Lookup ---
    auto t_look_0 = std::chrono::high_resolution_clock::now();
    auto e_it = ctx->exact_cache.find(e_key);
    auto t_look_1 = std::chrono::high_resolution_clock::now();
    ctx->total_lookup_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t_look_1 - t_look_0).count();

    if (e_it != ctx->exact_cache.end()) {
        ctx->num_algo_hits++;
        auto t_d0 = std::chrono::high_resolution_clock::now();
        status = cublasLtMatmul(ctx->ltHandle, e_it->second.opDesc, &alpha, d_A, e_it->second.Adesc, d_B, e_it->second.Bdesc, &beta, d_C, e_it->second.Cdesc, d_D, e_it->second.Ddesc, &e_it->second.algo, d_workspace, workspace_bytes, stream);
        
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "Error: cublasLtMatmul (cached) failed with status: " << cublasLtGetStatusName(status) << std::endl;
            std::cerr.flush();
            return -1;
        }
        auto t_d1 = std::chrono::high_resolution_clock::now();
        ctx->total_driver_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t_d1 - t_d0).count();
        ctx->total_cpp_duration_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t_d1 - start_total).count();
        return 0;
    }

    // --- 3. Miss: Setup & Heuristic ---
    ExactCacheEntry entry;
    status = cublasLtMatmulDescCreate(&entry.opDesc, get_compute_type((DispatchType)compute_dt), CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Error: cublasLtMatmulDescCreate failed with status: " << cublasLtGetStatusName(status) << std::endl;
        std::cerr.flush();
        return -1;
    }
    
    // Set Transpose Attributes
    cublasOperation_t opA = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    if (cublasLtMatmulDescSetAttribute(entry.opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Error: setting TRANSA failed with status: " << cublasLtGetStatusName(status) << std::endl; std::cerr.flush();
        return -1;
    }
    if (cublasLtMatmulDescSetAttribute(entry.opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Error: setting TRANSB failed with status: " << cublasLtGetStatusName(status) << std::endl; std::cerr.flush();
        return -1;
    }

    int rows_A = trans_a ? K : M; 
    int cols_A = trans_a ? M : K; 
    int rows_B = trans_b ? N : K; 
    int cols_B = trans_b ? K : N; 

    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;

    auto make_layout = [&](cublasLtMatrixLayout_t* l, DispatchType dt, int r, int c, int ld) -> cublasStatus_t {
        cublasStatus_t s = cublasLtMatrixLayoutCreate(l, get_cuda_dtype(dt), r, c, ld);
        if (s != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "Error: cublasLtMatrixLayoutCreate failed with status: " << cublasLtGetStatusName(s) << std::endl;
            std::cerr.flush(); // FORCE FLUSH
            return s;
        }
        s = cublasLtMatrixLayoutSetAttribute(*l, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
        if (s != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "Error: cublasLtMatrixLayoutSetAttribute failed with status: " << cublasLtGetStatusName(s) << std::endl;
            std::cerr.flush(); // FORCE FLUSH
            return s;
        }
        return CUBLAS_STATUS_SUCCESS;
    };

    if (make_layout(&entry.Adesc, (DispatchType)a_dt, rows_A, cols_A, lda) != CUBLAS_STATUS_SUCCESS) return -1;
    if (make_layout(&entry.Bdesc, (DispatchType)b_dt, rows_B, cols_B, ldb) != CUBLAS_STATUS_SUCCESS) return -1;
    if (make_layout(&entry.Cdesc, (DispatchType)c_dt, M, N, ldc) != CUBLAS_STATUS_SUCCESS) return -1;
    if (make_layout(&entry.Ddesc, (DispatchType)d_dt, M, N, ldd) != CUBLAS_STATUS_SUCCESS) return -1;

    // Fuzzy Cache Key logic
    int m_r = round_dim(M, ctx->round_multiple);
    int n_r = round_dim(N, ctx->round_multiple);
    int k_r = round_dim(K, ctx->round_multiple);
    FuzzyKey f_key = {m_r, n_r, k_r, (DispatchType)a_dt, (DispatchType)b_dt, (DispatchType)c_dt, (DispatchType)d_dt, (DispatchType)compute_dt, trans_a, trans_b};
    
    auto f_it = ctx->fuzzy_cache.find(f_key);
    if (f_it != ctx->fuzzy_cache.end()) {
        entry.algo = f_it->second;
        ctx->num_algo_hits++;
    } else {
        ctx->num_algos_saved++;
        cublasLtMatmulPreference_t pref;
        status = cublasLtMatmulPreferenceCreate(&pref);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "Error: cublasLtMatmulPreferenceCreate failed with status: " << cublasLtGetStatusName(status) << std::endl;
            std::cerr.flush();
            return -1;
        }
        uint64_t ws = workspace_bytes;
        status = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "Error: cublasLtMatmulPreferenceSetAttribute failed with status: " << cublasLtGetStatusName(status) << std::endl;
            std::cerr.flush();
            return -1;
        }

        std::vector<cublasLtMatmulHeuristicResult_t> heuristics(ctx->max_algos);
        int ret = 0;
        status = cublasLtMatmulAlgoGetHeuristic(
            ctx->ltHandle, 
            entry.opDesc, entry.Adesc, entry.Bdesc, entry.Cdesc, entry.Ddesc, 
            pref, 
            ctx->max_algos, 
            heuristics.data(), 
            &ret
        );
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "Error: cublasLtMatmulAlgoGetHeuristic failed with status: " << cublasLtGetStatusName(status) << std::endl;
            std::cerr.flush();
            return -1;
        }
        cublasLtMatmulPreferenceDestroy(pref); 
        
        if (ret == 0) {
            std::cerr << "Error: cublasLtMatmulAlgoGetHeuristic returned 0 algos" << std::endl;
            std::cerr.flush();
            return -1; 
        }

        entry.algo = heuristics[0].algo;
        ctx->fuzzy_cache[f_key] = entry.algo;
    }

    ctx->exact_cache[e_key] = entry;

    auto t_d0 = std::chrono::high_resolution_clock::now();
    status = cublasLtMatmul(ctx->ltHandle, entry.opDesc, &alpha, d_A, entry.Adesc, d_B, entry.Bdesc, &beta, d_C, entry.Cdesc, d_D, entry.Ddesc, &entry.algo, d_workspace, workspace_bytes, stream);
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Error: cublasLtMatmul (fresh) failed with status: " << cublasLtGetStatusName(status) << std::endl;
        std::cerr.flush(); // FORCE FLUSH
        return -1;
    }
    auto t_d1 = std::chrono::high_resolution_clock::now();
    ctx->total_driver_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t_d1 - t_d0).count();
    ctx->total_cpp_duration_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t_d1 - start_total).count();

    return 0;
}