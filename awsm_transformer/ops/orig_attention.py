import torch

class FlashAttentionNotAvailableError(Exception):
    """Raised when neither Flash Attention 2 nor Flash Attention 3 is available"""
    pass

try:
    from flash_attn_interface import flash_attn_3_cuda as flash3_gpu

    #flash3._flash_attn_fwd, flash3._flash_attn_bwd

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "h200" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except Exception as e:
    print(f"Flash 3 not available. Exception: {e}")
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn_2_cuda as flash2_gpu
    # flash2_gpu.varlen_fwd, flash2_gpu.varlen_bwd
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

if not FLASH_ATTN_2_AVAILABLE and not FLASH_ATTN_3_AVAILABLE:
    raise FlashAttentionNotAvailableError("Flash Attention 2 and Flash Attention 3 are not available")

def flash2_attention_fwd(q, k, v, out, softmax_lse, q_seq_offsets, k_seq_offsets, q_seq_lens, k_seq_lens, max_seqlen_q, max_seqlen_k, causal=True, window_size=(-1, -1), 
                            leftpad_k = None, block_table=None,alibi_slopes = None, dropout_p=0.0, softcap = 0.0, deterministic=True):
    ## mu-p uses 1 / d instead of 1 / d^2
    softmax_scale = q.shape[-1] ** -0.5
    gpu_out, gpu_softmax_lse, gpu_S_dmask, gpu_rng_state = flash2_gpu.varlen_fwd(q, k, v, out, 
                                                                    q_seq_offsets, k_seq_offsets, k_seq_lens, 
                                                                    leftpad_k, block_table, alibi_slopes, 
                                                                    max_seqlen_q, max_seqlen_k,
                                                                    dropout_p, softmax_scale, False,
                                                                    causal, window_size[0], window_size[1], 
                                                                    softcap, False, None)

    softmax_lse.copy_(gpu_softmax_lse)
    del gpu_softmax_lse
    return out, softmax_lse


def flash3_attention_fwd(q, k, v, out, softmax_lse, q_seq_offsets, k_seq_offsets, q_seq_lens, k_seq_lens, max_seqlen_q, max_seqlen_k, causal=True, window_size=(-1, -1), 
                            leftpad_k = None, block_table=None, alibi_slopes = None, dropout_p=0.0, softcap = 0.0, deterministic=True, sm_margin=0, attention_chunk=0,
                            rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None):
    ## mu-p uses 1 / d instead of 1 / d^2
    softmax_scale = q.shape[-1] ** -0.5
    gpu_out, gpu_softmax_lse, *rest = flash3_gpu.fwd(q, k, v, 
                                                        None, None, None, # k_new, v_new, qv 
                                                        out, 
                                                        q_seq_offsets, k_seq_offsets, None, # cu_seqlens_k_new
                                                        q_seq_lens, k_seq_lens, 
                                                        max_seqlen_q, max_seqlen_k, 
                                                        None, None, None, # page_table, kv_batch_idx, leftpad_k
                                                        None, None, None, # rotary cos, rotary sin, seqlens_rotary
                                                        None, None, None, # q_descale, k_descale, v_descale
                                                        softmax_scale, causal,
                                                        window_size[0], window_size[1],
                                                        attention_chunk, softcap, 
                                                        rotary_interleaved, 
                                                        scheduler_metadata,
                                                        num_splits, pack_gqa, 
                                                        sm_margin)
    softmax_lse.copy_(gpu_softmax_lse)
    return out, softmax_lse


def flash2_attention_bwd(dout, q, k, v, out, softmax_lse, dq, dk, dv, q_seq_offsets, k_seq_offsets, q_seq_lens, k_seq_lens, max_seqlen_q, max_seqlen_k, causal=True, window_size=(-1, -1), 
                                alibi_slopes = None, dropout_p=0.0, softcap = 0.0, deterministic=True, rng_state=None):
    ## mu-p uses 1 / d instead of 1 / d^2
    softmax_scale = q.shape[-1] ** -0.5
    
    # Create a contiguous variable. 
    # If softmax_lse was already contiguous, softmax_lse_contig IS softmax_lse (same pointer).
    # If it wasn't, softmax_lse_contig is a NEW tensor (new pointer).
    
    # Silently produces incorrect results if this isn't true
    softmax_lse_contig = softmax_lse.contiguous()
    

    dq, dk, dv, softmax_d = flash2_gpu.varlen_bwd(
                          dout, q, k, v, out, softmax_lse_contig, 
                          dq, dk, dv, 
                          q_seq_offsets, k_seq_offsets,
                          alibi_slopes,
                          max_seqlen_q, max_seqlen_k,
                          dropout_p, softmax_scale,
                          False, causal, window_size[0], window_size[1],
                          softcap, deterministic, None, rng_state)
    
    del softmax_d

    # LOGIC: Only delete if it's a different object (meaning a new copy was made)
    if softmax_lse_contig is not softmax_lse:
        del softmax_lse_contig

    return dq, dk, dv


def flash3_attention_bwd(dout, q, k, v, out, softmax_lse, dq, dk, dv, q_seq_offsets, k_seq_offsets, q_seq_lens, k_seq_lens, max_seqlen_q, max_seqlen_k, causal=True, window_size=(-1, -1), 
                               softcap = 0.0, deterministic=True, sm_margin=0):
    
    ## mu-p uses 1 / d instead of 1 / d^2
    softmax_scale = q.shape[-1] ** -0.5
    
    # Create a contiguous variable. 
    # If softmax_lse was already contiguous, softmax_lse_contig IS softmax_lse (same pointer).
    # If it wasn't, softmax_lse_contig is a NEW tensor (new pointer).
    
    ## Silently produces incorrect results if this isn't true
    softmax_lse_contig = softmax_lse.contiguous()
    
    dq, dk, dv, softmax_d, *rest = flash3_gpu.bwd(
                          dout, 
                          q, k, v, 
                          out, softmax_lse_contig,
                          dq, dk, dv, 
                          q_seq_offsets, k_seq_offsets,
                          None, None, # seqused q, seqused k
                          max_seqlen_q, max_seqlen_k, 
                          softmax_scale, causal,
                          window_size[0], window_size[1],
                          softcap, 
                          deterministic, 
                          sm_margin)
    
    del softmax_d

    # LOGIC: Only delete if it's a different object (meaning a new copy was made)
    if softmax_lse_contig is not softmax_lse:
        del softmax_lse_contig

    return dq, dk, dv

    
def awsm_attention_fwd(q, k, v, out, softmax_lse, q_seq_offsets, k_seq_offsets, q_seq_lens, k_seq_lens, max_seqlen_q, max_seqlen_k, causal=True, window_size=(-1, -1), sm_margin=0):
    
    # q: (total tokens, n_q_heads, head_dim)
    # k: (total tokens, n_kv_heads, head_dim)
    # v: (total tokens, n_kv_heads, head_dim)
    # out: (total tokens, n_q_heads, head_dim)
    # softmax_lse: (total tokens, n_q_heads)
    # q_seq_offsets (cumsum of sequence lengths): (num_seqs + 1)
    # k_seq_offsets (cumsum of sequence lengths): (num_seqs + 1)
    # q_seq_lens (sequence lengths): (num_seqs)
    # k_seq_lens (sequence lengths): (num_seqs)
    # max_seqlen_q: max sequence length of q
    # max_seqlen_k: max sequence length of k
    # causal: whether to use causal attention
    # window_size_left: window size left
    # window_size_right: window size right
    
    if FLASH_ATTN_3_AVAILABLE:
        return flash3_attention_fwd(q, k, v, out, softmax_lse, q_seq_offsets, k_seq_offsets, q_seq_lens, k_seq_lens, max_seqlen_q, max_seqlen_k, causal=causal, window_size=window_size, sm_margin=sm_margin)
    elif FLASH_ATTN_2_AVAILABLE:
        return flash2_attention_fwd(q, k, v, out, softmax_lse, q_seq_offsets, k_seq_offsets, q_seq_lens, k_seq_lens, max_seqlen_q, max_seqlen_k, causal=causal, window_size=window_size)
    else:
        raise FlashAttentionNotAvailableError(
            "Neither Flash Attention 2 nor Flash Attention 3 is available. "
            "Please install flash-attn package or ensure you have a compatible GPU (H100 for Flash Attention 3)."
        )

def awsm_attention_bwd(dout, q, k, v, out, softmax_lse, dq, dk, dv, q_seq_offsets, k_seq_offsets, q_seq_lens, k_seq_lens, max_seqlen_q, max_seqlen_k, causal=True, window_size=(-1, -1), 
                        deterministic=True, sm_margin=0):
    
    # dout: (total tokens, n_q_heads, head_dim)
    # q: (total tokens, n_q_heads, head_dim)
    # k: (total tokens, n_kv_heads, head_dim)
    # v: (total tokens, n_kv_heads, head_dim)
    # out: (total tokens, n_q_heads, head_dim)
    # softmax_lse: (total tokens, n_q_heads)
    # dq: (total tokens, n_q_heads, head_dim)
    # dk: (total tokens, n_kv_heads, head_dim)
    # dv: (total tokens, n_kv_heads, head_dim)
    # q_seq_offsets (cumsum of sequence lengths): (num_seqs + 1)
    # k_seq_offsets (cumsum of sequence lengths): (num_seqs + 1)
    # q_seq_lens (sequence lengths): (num_seqs)
    # k_seq_lens (sequence lengths): (num_seqs)
    # max_seqlen_q: max sequence length of q
    # max_seqlen_k: max sequence length of k
    # causal: whether to use causal attention
    # window_size_left: window size left
    # window_size_right: window size right

    if FLASH_ATTN_3_AVAILABLE:
        return flash3_attention_bwd(dout, q, k, v, out, softmax_lse, dq, dk, dv, q_seq_offsets, k_seq_offsets, q_seq_lens, k_seq_lens, max_seqlen_q, max_seqlen_k, causal=causal, window_size=window_size, deterministic=deterministic, sm_margin=sm_margin)
    elif FLASH_ATTN_2_AVAILABLE:
        return flash2_attention_bwd(dout, q, k, v, out, softmax_lse, dq, dk, dv, q_seq_offsets, k_seq_offsets, q_seq_lens, k_seq_lens, max_seqlen_q, max_seqlen_k, causal=causal, window_size=window_size, deterministic=deterministic)
    else:
        raise FlashAttentionNotAvailableError(
            "Neither Flash Attention 2 nor Flash Attention 3 is available. "
            "Please install flash-attn package or ensure you have a compatible GPU (H100 for Flash Attention 3)."
        )
