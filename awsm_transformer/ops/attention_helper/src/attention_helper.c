#include "attention_helper.h"

int flash_attention_get_workspace_size(int arch, int sm_count, FlashDtype flash_dtype, int is_training, int num_q_heads, int num_kv_heads, int head_dim, int max_chunk_size, int max_seq_len, int max_seqs_in_chunk, int is_causal, uint64_t * ret_workspace_size) {

	int ret;

	uint64_t fwd_workspace_size;

	int flash_dtype_as_int = (int) flash_dtype;

	if ((arch == 90 && USE_FLASH3_HOPPER) || ((arch == 80 || arch == 86 || arch == 89) && USE_FLASH3_AMPERE)) {
		ret = flash3_get_fwd_workspace_size(flash_dtype_as_int, arch, sm_count, num_q_heads, num_kv_heads, head_dim, max_chunk_size, max_seq_len, max_seqs_in_chunk, is_causal, &fwd_workspace_size);
	} else {
		ret = flash2_get_fwd_workspace_size(flash_dtype_as_int, arch, sm_count, num_q_heads, num_kv_heads, head_dim, max_chunk_size, max_seq_len, max_seqs_in_chunk, is_causal, &fwd_workspace_size);
	}

	if (ret){
		fprintf(stderr, "Error: unable to get flash attention workspace size\n");
		return -1;
	}

	if (!is_training){
		*ret_workspace_size = fwd_workspace_size;
		return 0;
	}

	uint64_t bwd_workspace_size;

	if ((arch == 90 && USE_FLASH3_HOPPER) || ((arch == 80 || arch == 86 || arch == 89) && USE_FLASH3_AMPERE)) {
		ret = flash3_get_bwd_workspace_size(flash_dtype_as_int, arch, sm_count, num_q_heads, num_kv_heads, head_dim, max_chunk_size, max_seq_len, max_seqs_in_chunk, is_causal, &bwd_workspace_size);
	} else {
		ret = flash2_get_bwd_workspace_size(flash_dtype_as_int, arch, sm_count, num_q_heads, num_kv_heads, head_dim, max_chunk_size, max_seq_len, max_seqs_in_chunk, is_causal, &bwd_workspace_size);
	}

	if (ret){
		fprintf(stderr, "Error: unable to get flash attention workspace size\n");
		return -1;
	}

	uint64_t max_workspace_size = MY_MAX(fwd_workspace_size, bwd_workspace_size);

	*ret_workspace_size = max_workspace_size;
	return 0;
}

int flash_attention_fwd(CUstream stream, int arch, int sm_count, FlashDtype flash_dtype, 
						int num_seqs, 
						int total_q, int total_k, 
						int * q_seq_offsets, int * q_seq_lens, int max_seqlen_q, 
						int * k_seq_offsets, int * k_seq_lens, int max_seqlen_k, 
						int num_q_heads, int num_kv_heads, int head_dim, 
						void * x_q, void * x_k, void * x_v, 
						void * x_attn_out, float * softmax_lse, 
						int is_causal, 
						uint64_t workspaceBytes, void * workspace) {

	int flash_dtype_as_int = (int) flash_dtype;


	// FLASH3 only supports SM80, SM86, SM89, SM90
	if ((arch == 90 && USE_FLASH3_HOPPER) || ((arch == 80 || arch == 86 || arch == 89) && USE_FLASH3_AMPERE)) {
		return flash3_fwd_wrapper(stream, arch, sm_count,
									flash_dtype_as_int,
									num_seqs, total_q, total_k,
									q_seq_offsets, q_seq_lens, max_seqlen_q,
									k_seq_offsets, k_seq_lens, max_seqlen_k,
									num_q_heads, num_kv_heads, head_dim,
									x_q, x_k, x_v,
									x_attn_out, softmax_lse,
									is_causal,
									workspaceBytes, workspace);
	}

	return flash2_fwd_wrapper(stream, arch, sm_count,
									flash_dtype_as_int,
									num_seqs, total_q, total_k,
									q_seq_offsets, q_seq_lens, max_seqlen_q,
									k_seq_offsets, k_seq_lens, max_seqlen_k,
									num_q_heads, num_kv_heads, head_dim,
									x_q, x_k, x_v,
									x_attn_out, softmax_lse,
									is_causal,
									workspaceBytes, workspace);
}


// inputs: same as fwd + dx_out (upstream gradient) and possibly different sized workspace

// purpose is to compute dx_q, dx_k, dx_v
int flash_attention_bwd(CUstream stream, int arch, int sm_count, FlashDtype flash_dtype, 
							int num_seqs, 
							int total_q, int total_k, 
							int * q_seq_offsets, int * q_seq_lens, int max_seqlen_q, 
							int * k_seq_offsets, int * k_seq_lens, int max_seqlen_k, 
							int num_q_heads, int num_kv_heads, int head_dim, 
							void * x_q, void * x_k, void * x_v, 
							void * x_attn_out, float * softmax_lse, 
							void * dx_out, 
							void * dx_q, void * dx_k, void * dx_v, 
							int is_causal, 
							uint64_t workspaceBytes, void * workspace) {

	int flash_dtype_as_int = (int) flash_dtype;

	// FLASH3 only supports SM80, SM86, SM89, SM90
	if ((arch == 90 && USE_FLASH3_HOPPER) || ((arch == 80 || arch == 86 || arch == 89) && USE_FLASH3_AMPERE)) {
		return flash3_bwd_wrapper(stream, arch, sm_count,
									flash_dtype_as_int,
									num_seqs, total_q, total_k,
									q_seq_offsets, q_seq_lens, max_seqlen_q,
									k_seq_offsets, k_seq_lens, max_seqlen_k,
									num_q_heads, num_kv_heads, head_dim,
									x_q, x_k, x_v,
									x_attn_out, softmax_lse,
									dx_out,
									dx_q, dx_k, dx_v,
									is_causal,
									workspaceBytes, workspace);
	} 

	return flash2_bwd_wrapper(stream, arch, sm_count,
									flash_dtype_as_int,
									num_seqs, total_q, total_k,
									q_seq_offsets, q_seq_lens, max_seqlen_q,
									k_seq_offsets, k_seq_lens, max_seqlen_k,
									num_q_heads, num_kv_heads, head_dim,
									x_q, x_k, x_v,
									x_attn_out, softmax_lse,
									dx_out,
									dx_q, dx_k, dx_v,
									is_causal,
									workspaceBytes, workspace);
}