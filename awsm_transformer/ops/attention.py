import torch
from awsm_attention import FlashAttentionHelper

helper = FlashAttentionHelper(device=torch.device("cuda:0"))

    
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
    
    helper.forward(q, k, v, out, softmax_lse, q_seq_offsets, k_seq_offsets, q_seq_lens, k_seq_lens, max_seqlen_q, max_seqlen_k, causal=causal)
    return out, softmax_lse

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

    helper.backward(
        dout,          # (total_q, n_q_heads, head_dim)
        q,             # (total_q, n_q_heads, head_dim)
        k,             # (total_k, n_kv_heads, head_dim)
        v,             # (total_k, n_kv_heads, head_dim)
        out,           # (total_q, n_q_heads, head_dim)
        softmax_lse,   # (total_q, n_q_heads)
        dq,            # (total_q, n_q_heads, head_dim)
        dk,            # (total_k, n_kv_heads, head_dim)
        dv,            # (total_k, n_kv_heads, head_dim)
        q_seq_offsets, # (num_seqs + 1,) int32
        k_seq_offsets, # (num_seqs + 1,) int32
        q_seq_lens,     # (num_seqs,) int32
        k_seq_lens,     # (num_seqs,) int32
        max_seqlen_q,
        max_seqlen_k,
        causal,
    )

    return dq, dk, dv
