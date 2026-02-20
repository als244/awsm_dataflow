import torch

import numpy as np

from .ops import awsm_rmsnorm_fwd, awsm_rmsnorm_fwd_recompute, awsm_rmsnorm_bwd, awsm_rope_fwd, awsm_rope_bwd, awsm_attention_fwd, awsm_swiglu_fwd, awsm_attention_bwd, awsm_rmsnorm_bwd, awsm_swiglu_bwd, awsm_adamw_step, awsm_muon_step

from .matmul_dispatchers import dispatcher

from .utils import get_torch_dtype
from .mem_register import pin_tensor

class TransformerLayer():

    def __init__(self, layer_id, model_dims, model_hyperparams, is_muon=False, secondary_compute_stream=None):
        self.layer_id = layer_id
        self.model_dims = model_dims
        self.model_hyperparams = model_hyperparams
        self.step_num = 0
        self.is_muon = is_muon
        self.secondary_compute_stream = secondary_compute_stream
        self.max_saved_activations_level = 3
    
    def forward(self, X, chunk_metadata, weights, base_act_slot, fwd_context):

        num_tokens = X.shape[0]

        act_slot = {}

        ## use view of base act slot with the correct shape for this chunk
        for k, v in base_act_slot.items():
            if k != "softmax_lse" and v.shape[0] != num_tokens:
                act_slot[k] = v[:num_tokens, :]
            elif k == "softmax_lse" and v.shape[1] != num_tokens:
                act_slot[k] = v[:, :num_tokens]
            else:
                act_slot[k] = v

        ## copy x to act_slot["x_inp"]
        act_slot["x_inp"].copy_(X)

        x_temp = torch.empty(X.shape, dtype=X.dtype, device=X.device)

        # Attention part
        attn_norm_output, attn_norm_rstd = awsm_rmsnorm_fwd(X, W=weights["w_attn_norm"], output=x_temp, rstd=act_slot["attn_norm_rstd"], rms_norm_eps=self.model_hyperparams["rms_norm_eps"])

        head_dim = act_slot["xq"].shape[-1]
        n_heads = act_slot["xq"].shape[1]
        n_kv_heads = act_slot["xk"].shape[1]

        xq = torch.matmul(attn_norm_output, weights["w_q"], out=act_slot["xq"].view(-1, n_heads * head_dim))
        xk = torch.matmul(attn_norm_output, weights["w_k"], out=act_slot["xk"].view(-1, n_kv_heads * head_dim))
        xv = torch.matmul(attn_norm_output, weights["w_v"], out=act_slot["xv"].view(-1, n_kv_heads * head_dim))
        
        rope_q, rope_k = awsm_rope_fwd([xq.view(-1, n_heads, head_dim), xk.view(-1, n_kv_heads, head_dim)], chunk_metadata["seq_positions"], self.model_hyperparams["position_angles"])

        cur_seq_offset = 0

        ## this is being explicity for readability, but likely can just copy whole chunk
        """
        for seq_idx in range(len(chunk_metadata["seq_lens_host"])):
            prior_start_idx = chunk_metadata["prior_seq_offsets_host"][seq_idx]
            prior_end_idx = prior_start_idx + chunk_metadata["prior_seq_lens_host"][seq_idx]

            new_seqlen = chunk_metadata["seq_lens_host"][seq_idx]
            new_end_idx = prior_end_idx + new_seqlen

            ## copy rope_k and xv to context windows...
            fwd_context["k"][prior_end_idx:new_end_idx, :].copy_(rope_k[cur_seq_offset:cur_seq_offset + new_seqlen, :])
            fwd_context["v"][prior_end_idx:new_end_idx, :].copy_(xv.view(-1, n_kv_heads, head_dim)[cur_seq_offset:cur_seq_offset + new_seqlen, :])

            cur_seq_offset += new_seqlen
        """

        start_chunk_idx = chunk_metadata["prior_seq_offsets_host"][0] + chunk_metadata["prior_seq_lens_host"][0]
        total_q = chunk_metadata["total_q"]
        fwd_context["k"][start_chunk_idx: start_chunk_idx + total_q, :].copy_(rope_k)
        fwd_context["v"][start_chunk_idx: start_chunk_idx + total_q, :].copy_(xv.view(-1, n_kv_heads, head_dim))

        total_k = chunk_metadata["total_k"]

        attn_result, softmax_lse = awsm_attention_fwd(rope_q.view(-1, n_heads, head_dim), fwd_context["k"][:total_k, :], fwd_context["v"][:total_k, :],
                                    act_slot["attn_result"], act_slot["softmax_lse"], 
                                    chunk_metadata["q_seq_offsets"], chunk_metadata["k_seq_offsets"],
                                    chunk_metadata["q_seq_lens"], chunk_metadata["k_seq_lens"], 
                                    chunk_metadata["max_seqlen_q"], chunk_metadata["max_seqlen_k"],
                                    causal=self.model_dims["is_causal"], window_size=(self.model_hyperparams["window_size_left"], self.model_hyperparams["window_size_right"]))

        ## Have input == output to avoid implicity PyTorch DtoD copy

        cur_stream_ptr = torch.cuda.current_stream().cuda_stream
        attn_output_with_residual = dispatcher.matmul(cur_stream_ptr, A=attn_result.view(-1, n_heads * head_dim), B=weights["w_o"], C=X, D=act_slot["xo"], alpha=1.0, beta=1.0)

        # MLP part

        ffn_norm_output, ffn_norm_rstd = awsm_rmsnorm_fwd(attn_output_with_residual.view(-1, n_heads * head_dim), W=weights["w_ffn_norm"], output=x_temp, rstd=act_slot["ffn_norm_rstd"], rms_norm_eps=self.model_hyperparams["rms_norm_eps"])
        
        x1 = torch.matmul(ffn_norm_output, weights["w_1"], out=act_slot["x1"])
        x3 = torch.matmul(ffn_norm_output, weights["w_3"], out=act_slot["x3"])

        x_temp_mlp = torch.empty(x1.shape, dtype=x1.dtype, device=x1.device)
        
        swiglu_result = awsm_swiglu_fwd(x1, x3, out=x_temp_mlp)

        ## Now input == output so avoid implicity DtoD copy
        layer_output = torch.addmm(attn_output_with_residual, swiglu_result, weights["w_2"], out=X)

        del x_temp
        del x_temp_mlp

        return layer_output, act_slot
    

    ### Always called before bwd pass and optionally recomputes dependening on contents of fwd_act_slot
    ### Assumes that values in fwd_context are already populated correctly...!
    def forward_recompute(self, fwd_act_slot, base_act_slot, chunk_metadata, weights, fwd_context):

        ## require input to be saved
        X_inp = fwd_act_slot["x_inp"]

        num_tokens = X_inp.shape[0]


        ## require xk and xv (local) to be saved (post position embed)

        xk = fwd_act_slot["xk"]
        xv = fwd_act_slot["xv"]

        n_heads = self.model_dims["n_heads"]
        head_dim = self.model_dims["head_dim"]

        if "xq" not in fwd_act_slot:
            fwd_act_slot["xq"] = base_act_slot["xq"][:num_tokens, :].view(num_tokens, -1)
            fwd_act_slot["attn_norm_output"] = awsm_rmsnorm_fwd_recompute(X_inp, weights["w_attn_norm"], fwd_act_slot["attn_norm_rstd"])
            xq = torch.matmul(fwd_act_slot["attn_norm_output"], weights["w_q"], out=fwd_act_slot["xq"])
            fwd_act_slot["xq"] = xq.view(-1, n_heads, head_dim)
            rope_q = awsm_rope_fwd([fwd_act_slot["xq"].view(-1, n_heads, head_dim)], chunk_metadata["seq_positions"], self.model_hyperparams["position_angles"])[0]
        else:
            rope_q = fwd_act_slot["xq"].view(-1, n_heads, head_dim)

        if "attn_result" not in fwd_act_slot:

            fwd_act_slot["attn_result"] = base_act_slot["attn_result"][:num_tokens, :]
            fwd_act_slot["softmax_lse"] = base_act_slot["softmax_lse"][:, :num_tokens]

            attn_result, softmax_lse = awsm_attention_fwd(rope_q.view(-1, n_heads, head_dim), fwd_context["k"][:num_tokens, :], fwd_context["v"][:num_tokens, :],
                                    fwd_act_slot["attn_result"], fwd_act_slot["softmax_lse"], 
                                    chunk_metadata["q_seq_offsets"], chunk_metadata["k_seq_offsets"],
                                    chunk_metadata["q_seq_lens"], chunk_metadata["k_seq_lens"], 
                                    chunk_metadata["max_seqlen_q"], chunk_metadata["max_seqlen_k"],
                                    causal=self.model_dims["is_causal"], window_size=(self.model_hyperparams["window_size_left"], self.model_hyperparams["window_size_right"]))
        else:   
            attn_result = fwd_act_slot["attn_result"]
        

        if "xo" not in fwd_act_slot:

            fwd_act_slot["xo"] = base_act_slot["xo"][:num_tokens, :]
            attn_output_with_residual = torch.addmm(X_inp, attn_result.view(-1, n_heads * head_dim), weights["w_o"], out=fwd_act_slot["xo"])
        else:
            attn_output_with_residual = fwd_act_slot["xo"]

        
        if "x1" not in fwd_act_slot or "x3" not in fwd_act_slot:

            recompute_x1 = False
            recompute_x3 = False

            if "x1" not in fwd_act_slot:
                fwd_act_slot["x1"] = base_act_slot["x1"][:num_tokens, :]
                recompute_x1 = True
            if "x3" not in fwd_act_slot:
                fwd_act_slot["x3"] = base_act_slot["x3"][:num_tokens, :]
                recompute_x3 = True
            fwd_act_slot["ffn_norm_output"] = awsm_rmsnorm_fwd_recompute(attn_output_with_residual.view(-1, n_heads * head_dim), weights["w_ffn_norm"], fwd_act_slot["ffn_norm_rstd"])
            if recompute_x1:
                torch.matmul(fwd_act_slot["ffn_norm_output"], weights["w_1"], out=fwd_act_slot["x1"])
            if recompute_x3:
                torch.matmul(fwd_act_slot["ffn_norm_output"], weights["w_3"], out=fwd_act_slot["x3"])
        
        return fwd_act_slot
    

    def backward(self, dX, chunk_metadata, weights, grad_weights, fwd_act_slot, fwd_context, bwd_context, total_tokens_per_step=None):
        
        #torch.cuda.synchronize()
        
        #print(f"Layer {self.layer_id}\n\tUpstream Gradients: {dX}\n\n\n", flush=True)
        ### Part 1. MLP

        ## a.) do weight gradient upddates for w2, w1, and w3
        ## b.) get upstream gradient for attention block, which is the downstream gradient of ffn norm norm

        # 1.) backwards through output projection
        dX_up_act = torch.matmul(dX, weights["w_2"].T)

        # 2.) backwards through swiglu and recompute forward activations
        dx1_up, dx3_up, fwd_act_swiglu = awsm_swiglu_bwd(fwd_act_slot["x1"], fwd_act_slot["x3"], dX_up_act, store_activations=True)

        # 3.) now do model gradient computation for dW2
        torch.addmm(grad_weights["g_2"], fwd_act_swiglu.T, dX, alpha=1.0, beta=1.0, out=grad_weights["g_2"])

        del fwd_act_swiglu
        del dX_up_act

        # 4.) now backprop through w1 and w3 and accumulate result into upstream gradient of ffn norm

        dX_ffn_norm_up = torch.matmul(dx1_up, weights["w_1"].T)
        torch.addmm(dX_ffn_norm_up, dx3_up, weights["w_3"].T, alpha=1.0, beta=1.0, out=dX_ffn_norm_up)

        # 5.) now backprop through ffn norm
        ### accumulate result into residual gradient stream (original dX)...
        if "ffn_norm_output" in fwd_act_slot:
            dX_ffn_norm_down, dW_ffn_norm, _ = awsm_rmsnorm_bwd(dX_ffn_norm_up, fwd_act_slot["xo"], weights["w_ffn_norm"], fwd_act_slot["ffn_norm_rstd"], dW = grad_weights["g_ffn_norm"], dX = dX, recompute_output=False)
            ffn_norm_fwd_output = fwd_act_slot.pop("ffn_norm_output")
        else:
            dX_ffn_norm_down, dW_ffn_norm, ffn_norm_fwd_output = awsm_rmsnorm_bwd(dX_ffn_norm_up, fwd_act_slot["xo"], weights["w_ffn_norm"], fwd_act_slot["ffn_norm_rstd"], dW = grad_weights["g_ffn_norm"], dX = dX, recompute_output=True)

        del dX_ffn_norm_up

        # 6.) now do weight gradients for w1 and w3 now that we have inputs to matmuls (recomputed ffn norm output) and upstream gradients
        torch.addmm(grad_weights["g_1"], ffn_norm_fwd_output.T, dx1_up, alpha=1.0, beta=1.0, out=grad_weights["g_1"])
        torch.addmm(grad_weights["g_3"], ffn_norm_fwd_output.T, dx3_up, alpha=1.0, beta=1.0, out=grad_weights["g_3"])

        del dx1_up
        del dx3_up
        del ffn_norm_fwd_output



        ### Part 2. Attention

        ### using the new dX written above, we will now backprop through the attention block

        ## a.) do weight gradient upddates for w0, wq, wk, wv
        ## b.) accumulate gradients into bwd context for prior chunks (lower seq positions)
        ## c.) return downstream gradient of attn norm which will be used as upstream gradient for next downstream block

        num_tokens = dX.shape[0]
        n_heads = self.model_dims["n_heads"]
        n_kv_heads = self.model_dims["n_kv_heads"]
        head_dim = self.model_dims["head_dim"]

        attn_result = fwd_act_slot["attn_result"].view(num_tokens, -1)
        

        # 1.) Update weight gradient for wo
        torch.addmm(grad_weights["g_o"], attn_result.T, dX, alpha=1.0, beta=1.0, out=grad_weights["g_o"])

        # 2.) backwards through output projection
        dX_up_attn = torch.matmul(dX, weights["w_o"].T)

        dX_up_attn = dX_up_attn.view(num_tokens, n_heads, head_dim)

        total_k = chunk_metadata["total_k"]

        # 2.) backwards attention
        dq = torch.zeros_like(dX_up_attn)

     

        dq, full_dk, full_dv = awsm_attention_bwd(dX_up_attn, fwd_act_slot["xq"].view(-1, n_heads, head_dim), fwd_context["k"][:total_k, :], fwd_context["v"][:total_k, :], fwd_act_slot["attn_result"], fwd_act_slot["softmax_lse"], 
                                                dq, bwd_context["dk"][:total_k, :], bwd_context["dv"][:total_k, :], 
                                                chunk_metadata["q_seq_offsets"], chunk_metadata["k_seq_offsets"], chunk_metadata["q_seq_lens"], chunk_metadata["k_seq_lens"], 
                                                chunk_metadata["max_seqlen_q"], chunk_metadata["max_seqlen_k"], 
                                                causal=self.model_dims["is_causal"], window_size=(self.model_hyperparams["window_size_left"], self.model_hyperparams["window_size_right"]))

        del dX_up_attn

        local_dk = torch.zeros_like(fwd_act_slot["xk"])
        local_dv = torch.zeros_like(fwd_act_slot["xv"])
        

        # 2b.) Now use the local (corresponding to this chunk) dK, dV for use in computed further downstream gradient
        """
        cur_seq_offset = 0
        for seq_idx in range(len(chunk_metadata["seq_lens_host"])):
            prior_start_idx = chunk_metadata["prior_seq_offsets_host"][seq_idx]
            prior_end_idx = prior_start_idx + chunk_metadata["prior_seq_lens_host"][seq_idx]

            new_seqlen = chunk_metadata["seq_lens_host"][seq_idx]
            new_end_idx = prior_end_idx + new_seqlen

            ## copy rope_k and xv to context windows...
            local_dk[cur_seq_offset:cur_seq_offset + new_seqlen].copy_(bwd_context["dk"][prior_end_idx:new_end_idx, :])
            local_dv[cur_seq_offset:cur_seq_offset + new_seqlen].copy_(bwd_context["dv"][prior_end_idx:new_end_idx, :])

            ## can zero out this position in bwd now...
            bwd_context["dk"][prior_end_idx:new_end_idx, :].zero_()
            bwd_context["dv"][prior_end_idx:new_end_idx, :].zero_()

            cur_seq_offset += new_seqlen
        """
        start_chunk_idx = chunk_metadata["prior_seq_offsets_host"][0] + chunk_metadata["prior_seq_lens_host"][0]
        total_q = chunk_metadata["total_q"]
        local_dk.copy_(bwd_context["dk"][start_chunk_idx: start_chunk_idx + total_q, :])
        local_dv.copy_(bwd_context["dv"][start_chunk_idx: start_chunk_idx + total_q, :])
        bwd_context["dk"][start_chunk_idx: start_chunk_idx + total_q, :].zero_()
        bwd_context["dv"][start_chunk_idx: start_chunk_idx + total_q, :].zero_()

        # 4.) rope bwd on dq and local_dk
        dq, local_dk = awsm_rope_bwd([dq.view(-1, n_heads, head_dim), local_dk.view(-1, n_kv_heads, head_dim)], chunk_metadata["seq_positions"], self.model_hyperparams["position_angles"])
        
        
        # 3.) backprop through wq, wk, wv and accumulate result into upstream gradient of attn norm

        dX_attn_norm_up = torch.matmul(dq.view(num_tokens, -1), weights["w_q"].T)
        torch.addmm(dX_attn_norm_up, local_dk.view(num_tokens, -1), weights["w_k"].T, alpha=1.0, beta=1.0, out=dX_attn_norm_up)
        torch.addmm(dX_attn_norm_up, local_dv.view(num_tokens, -1), weights["w_v"].T, alpha=1.0, beta=1.0, out=dX_attn_norm_up)

        # 4.) now backprop through attn norm, accumulating result into overwritten dX (already updated from MLP block)
        if "attn_norm_output" in fwd_act_slot:
            dX, dW_attn_norm, _ = awsm_rmsnorm_bwd(dX_attn_norm_up, fwd_act_slot["x_inp"], weights["w_attn_norm"], fwd_act_slot["attn_norm_rstd"], dW = grad_weights["g_attn_norm"], dX = dX,recompute_output=False)
            attn_norm_fwd_output = fwd_act_slot.pop("attn_norm_output")
        else:
            dX, dW_attn_norm, attn_norm_fwd_output = awsm_rmsnorm_bwd(dX_attn_norm_up, fwd_act_slot["x_inp"], weights["w_attn_norm"], fwd_act_slot["attn_norm_rstd"], dW = grad_weights["g_attn_norm"], dX = dX,recompute_output=True)        
       
        # 5.) update weight gradients for wq, wk, wv
        torch.addmm(grad_weights["g_v"], attn_norm_fwd_output.T, local_dv.view(num_tokens, -1), alpha=1.0, beta=1.0, out=grad_weights["g_v"])
        torch.addmm(grad_weights["g_k"], attn_norm_fwd_output.T, local_dk.view(num_tokens, -1), alpha=1.0, beta=1.0, out=grad_weights["g_k"])
        torch.addmm(grad_weights["g_q"], attn_norm_fwd_output.T, dq.view(num_tokens, -1), alpha=1.0, beta=1.0, out=grad_weights["g_q"])
       
            
        del attn_norm_fwd_output
        del dX_attn_norm_up
        del dq
        del local_dk
        del local_dv
        

        return dX

    def step(self, weights, grad_weights, opt_state, opt_hyperparams):

        if self.is_muon:
            return self.step_muon(weights, grad_weights, opt_state, opt_hyperparams)

        lr = opt_hyperparams["lr"]
        beta1 = opt_hyperparams["beta1"]
        beta2 = opt_hyperparams["beta2"]
        eps = opt_hyperparams["eps"]
        weight_decay = opt_hyperparams["weight_decay"]
        step_num = opt_hyperparams["step_num"]

        ret = awsm_adamw_step(weights["w_attn_norm"], grad_weights["g_attn_norm"], opt_state["o_m_attn_norm"], opt_state["o_v_attn_norm"], 
               lr=lr, beta1=beta1, beta2=beta2, eps=eps, 
               weight_decay=weight_decay, step=step_num)
        if ret != 0:
            print(f"AdamW Step failed for attn norm at layer {self.layer_id}")
            return -1
        ret = awsm_adamw_step(weights["w_q"], grad_weights["g_q"], opt_state["o_m_q"], opt_state["o_v_q"], 
               lr=lr, beta1=beta1, beta2=beta2, eps=eps, 
               weight_decay=weight_decay, step=step_num)
        if ret != 0:
            print(f"AdamW Step failed for q at layer {self.layer_id}")
            return -1
        ret = awsm_adamw_step(weights["w_k"], grad_weights["g_k"], opt_state["o_m_k"], opt_state["o_v_k"], 
               lr=lr, beta1=beta1, beta2=beta2, eps=eps, 
               weight_decay=weight_decay, step=step_num)
        if ret != 0:
            print(f"AdamW Step failed for k at layer {self.layer_id}")
            return -1
        ret = awsm_adamw_step(weights["w_v"], grad_weights["g_v"], opt_state["o_m_v"], opt_state["o_v_v"], 
               lr=lr, beta1=beta1, beta2=beta2, eps=eps, 
               weight_decay=weight_decay, step=step_num)
        if ret != 0:
            print(f"AdamW Step failed for v at layer {self.layer_id}")
            return -1
        ret = awsm_adamw_step(weights["w_o"], grad_weights["g_o"], opt_state["o_m_o"], opt_state["o_v_o"], 
               lr=lr, beta1=beta1, beta2=beta2, eps=eps, 
               weight_decay=weight_decay, step=step_num)
        if ret != 0:
            print(f"AdamW Step failed for o at layer {self.layer_id}")
            return -1
        ret = awsm_adamw_step(weights["w_ffn_norm"], grad_weights["g_ffn_norm"], opt_state["o_m_ffn_norm"], opt_state["o_v_ffn_norm"], 
               lr=lr, beta1=beta1, beta2=beta2, eps=eps, 
               weight_decay=weight_decay, step=step_num)
        if ret != 0:
            print(f"AdamW Step failed for ffn norm at layer {self.layer_id}")
            return -1
        ret = awsm_adamw_step(weights["w_1"], grad_weights["g_1"], opt_state["o_m_1"], opt_state["o_v_1"], 
               lr=lr, beta1=beta1, beta2=beta2, eps=eps, 
               weight_decay=weight_decay, step=step_num)
        if ret != 0:
            print(f"AdamW Step failed for w1 at layer {self.layer_id}")
            return -1
        ret = awsm_adamw_step(weights["w_3"], grad_weights["g_3"], opt_state["o_m_3"], opt_state["o_v_3"], 
               lr=lr, beta1=beta1, beta2=beta2, eps=eps, 
               weight_decay=weight_decay, step=step_num)
        if ret != 0:
            print(f"AdamW Step failed for w3 at layer {self.layer_id}")
            return -1
        ret = awsm_adamw_step(weights["w_2"], grad_weights["g_2"], opt_state["o_m_2"], opt_state["o_v_2"], 
               lr=lr, beta1=beta1, beta2=beta2, eps=eps, 
               weight_decay=weight_decay, step=step_num)
        if ret != 0:
            print(f"AdamW Step failed for w2 at layer {self.layer_id}")
            return -1
        
        return 0

    def step_muon(self, weights, grad_weights, opt_state, opt_hyperparams):

        lr = opt_hyperparams["lr"]
        beta1 = opt_hyperparams["beta1"]
        beta2 = opt_hyperparams["beta2"]
        muon_beta = opt_hyperparams["beta1"]
        eps = opt_hyperparams["eps"]
        weight_decay = opt_hyperparams["weight_decay"]
        step_num = opt_hyperparams["step_num"]


        ### Norms use AdamW

        ret = awsm_adamw_step(weights["w_attn_norm"], grad_weights["g_attn_norm"], opt_state["o_m_attn_norm"], opt_state["o_v_attn_norm"], 
               lr=lr, beta1=beta1, beta2=beta2, eps=eps, 
               weight_decay=weight_decay, step=step_num)
        if ret != 0:
            print(f"AdamW Step failed for attn norm at layer {self.layer_id}")
            return -1
        
        ret = awsm_adamw_step(weights["w_ffn_norm"], grad_weights["g_ffn_norm"], opt_state["o_m_ffn_norm"], opt_state["o_v_ffn_norm"], 
               lr=lr, beta1=beta1, beta2=beta2, eps=eps, 
               weight_decay=weight_decay, step=step_num)
        if ret != 0:
            print(f"AdamW Step failed for ffn norm at layer {self.layer_id}")
            return -1
    
        ### Do Muon Steps

        ret = awsm_muon_step(weights["w_q"], grad_weights["g_q"], opt_state["o_m_q"], 
               lr=lr, beta=muon_beta, eps=eps,weight_decay=weight_decay)
        if ret != 0:
            print(f"Muon Step failed for wq at layer {self.layer_id}")
            return -1
        ret = awsm_muon_step(weights["w_k"], grad_weights["g_k"], opt_state["o_m_k"], 
               lr=lr, beta=muon_beta, eps=eps,weight_decay=weight_decay)
        if ret != 0:
            print(f"Muon Step failed for wk at layer {self.layer_id}")
            return -1
        ret = awsm_muon_step(weights["w_v"], grad_weights["g_v"], opt_state["o_m_v"], 
               lr=lr, beta=muon_beta, eps=eps,weight_decay=weight_decay)
        if ret != 0:
            print(f"Muon Step failed for wv at layer {self.layer_id}")
            return -1
        ret = awsm_muon_step(weights["w_o"], grad_weights["g_o"], opt_state["o_m_o"], 
               lr=lr, beta=muon_beta, eps=eps,weight_decay=weight_decay)
        if ret != 0:
            print(f"Muon Step failed for wo at layer {self.layer_id}")
            return -1
        ret = awsm_muon_step(weights["w_1"], grad_weights["g_1"], opt_state["o_m_1"], 
               lr=lr, beta=muon_beta, eps=eps,weight_decay=weight_decay)
        if ret != 0:
            print(f"Muon Step failed for w1 at layer {self.layer_id}")
            return -1
        ret = awsm_muon_step(weights["w_3"], grad_weights["g_3"], opt_state["o_m_3"], 
               lr=lr, beta=muon_beta, eps=eps,weight_decay=weight_decay)
        if ret != 0:
            print(f"Muon Step failed for w3 at layer {self.layer_id}")
            return -1
        ret = awsm_muon_step(weights["w_2"], grad_weights["g_2"], opt_state["o_m_2"], 
               lr=lr, beta=muon_beta, eps=eps,weight_decay=weight_decay)
        if ret != 0:
            print(f"Muon Step failed for w2 at layer {self.layer_id}")
            return -1

        return 0
    
    def create(self, buffer = None, device = "cpu", pin_memory = True, is_grad=False, dtype_mapping = None):

        d_model = self.model_dims["d_model"]
        n_heads = self.model_dims["n_heads"]
        n_kv_heads = self.model_dims["n_kv_heads"]
        head_dim = self.model_dims["head_dim"]
        expert_dim = self.model_dims["expert_dim"]

        prefix = "w_"

        if is_grad:
            prefix = "g_"

        if dtype_mapping is None:
            dtype_mapping = {
                prefix + "attn_norm": torch.bfloat16,
                prefix + "q": torch.bfloat16,
                prefix + "k": torch.bfloat16,
                prefix + "v": torch.bfloat16,
                prefix + "o": torch.bfloat16,
                prefix + "ffn_norm": torch.bfloat16,
                prefix + "1": torch.bfloat16,
                prefix + "3": torch.bfloat16,
                prefix + "2": torch.bfloat16,
            }

        if device != "cpu":
            pin_memory = False

        

        if buffer is None:
            return {
                f"{prefix}attn_norm": pin_tensor(torch.zeros(d_model, device=device, dtype=dtype_mapping[prefix + "attn_norm"])),
                f"{prefix}q": pin_tensor(torch.zeros(d_model, n_heads * head_dim, device=device, dtype=dtype_mapping[prefix + "q"])),
                f"{prefix}k": pin_tensor(torch.zeros(d_model, n_kv_heads * head_dim, device=device, dtype=dtype_mapping[prefix + "k"])),
                f"{prefix}v": pin_tensor(torch.zeros(d_model, n_kv_heads * head_dim, device=device, dtype=dtype_mapping[prefix + "v"])),
                f"{prefix}o": pin_tensor(torch.zeros(n_heads * head_dim, d_model, device=device, dtype=dtype_mapping[prefix + "o"])),
                f"{prefix}ffn_norm": pin_tensor(torch.zeros(d_model, device=device, dtype=dtype_mapping[prefix + "ffn_norm"])),
                f"{prefix}1": pin_tensor(torch.zeros(d_model, expert_dim, device=device, dtype=dtype_mapping[prefix + "1"])),
                f"{prefix}3": pin_tensor(torch.zeros(d_model, expert_dim, device=device, dtype=dtype_mapping[prefix + "3"])),
                f"{prefix}2": pin_tensor(torch.zeros(expert_dim, d_model, device=device, dtype=dtype_mapping[prefix + "2"])),
            }
        else:

            layer = {}
            attn_norm_size = d_model * dtype_mapping[prefix + "attn_norm"].itemsize
            q_size = d_model * n_heads * head_dim * dtype_mapping[prefix + "q"].itemsize
            k_size = d_model * n_kv_heads * head_dim * dtype_mapping[prefix + "k"].itemsize
            v_size = d_model * n_kv_heads * head_dim * dtype_mapping[prefix + "v"].itemsize
            o_size = n_heads * head_dim * d_model * dtype_mapping[prefix + "o"].itemsize
            ffn_norm_size = d_model * dtype_mapping[prefix + "ffn_norm"].itemsize
            x1_size = d_model * expert_dim * dtype_mapping[prefix + "1"].itemsize
            x3_size = d_model * expert_dim * dtype_mapping[prefix + "3"].itemsize
            x2_size = expert_dim * d_model * dtype_mapping[prefix + "2"].itemsize

            cur_offset = 0
            layer[prefix + "attn_norm"] = buffer[:attn_norm_size].view(dtype_mapping[prefix + "attn_norm"]).reshape(d_model)
            cur_offset += attn_norm_size
            layer[prefix + "q"] = buffer[cur_offset:cur_offset + q_size].view(dtype_mapping[prefix + "q"]).reshape(d_model, n_heads * head_dim)
            cur_offset += q_size
            layer[prefix + "k"] = buffer[cur_offset:cur_offset + k_size].view(dtype_mapping[prefix + "k"]).reshape(d_model, n_kv_heads * head_dim)
            cur_offset += k_size
            layer[prefix + "v"] = buffer[cur_offset:cur_offset + v_size].view(dtype_mapping[prefix + "v"]).reshape(d_model, n_kv_heads * head_dim)
            cur_offset += v_size
            layer[prefix + "o"] = buffer[cur_offset:cur_offset + o_size].view(dtype_mapping[prefix + "o"]).reshape(n_heads * head_dim, d_model)
            cur_offset += o_size
            layer[prefix + "ffn_norm"] = buffer[cur_offset:cur_offset + ffn_norm_size].view(dtype_mapping[prefix + "ffn_norm"]).reshape(d_model)
            cur_offset += ffn_norm_size
            layer[prefix + "1"] = buffer[cur_offset:cur_offset + x1_size].view(dtype_mapping[prefix + "1"]).reshape(d_model, expert_dim)
            cur_offset += x1_size
            layer[prefix + "3"] = buffer[cur_offset:cur_offset + x3_size].view(dtype_mapping[prefix + "3"]).reshape(d_model, expert_dim)
            cur_offset += x3_size
            layer[prefix + "2"] = buffer[cur_offset:cur_offset + x2_size].view(dtype_mapping[prefix + "2"]).reshape(expert_dim, d_model)
            cur_offset += x2_size

            return layer

    def create_opt(self, buffer = None, device = "cpu", pin_memory = True, dtype_mapping = None):

        d_model = self.model_dims["d_model"]
        n_heads = self.model_dims["n_heads"]
        n_kv_heads = self.model_dims["n_kv_heads"]
        head_dim = self.model_dims["head_dim"]
        expert_dim = self.model_dims["expert_dim"]

        prefixes = ["o_m_", "o_v_"]

        if device != "cpu":
            pin_memory = False

        if dtype_mapping is None:
            dtype_mapping = {}

            for prefix in prefixes:
                dtype_mapping[prefix + "attn_norm"] = torch.bfloat16
                dtype_mapping[prefix + "q"] = torch.bfloat16
                dtype_mapping[prefix + "k"] = torch.bfloat16
                dtype_mapping[prefix + "v"] = torch.bfloat16
                dtype_mapping[prefix + "o"] = torch.bfloat16
                dtype_mapping[prefix + "ffn_norm"] = torch.bfloat16
                dtype_mapping[prefix + "1"] = torch.bfloat16
                dtype_mapping[prefix + "3"] = torch.bfloat16
                dtype_mapping[prefix + "2"] = torch.bfloat16

        opt_layer = {}     
        cur_offset = 0
        if buffer is None:
            for prefix in prefixes:
                opt_layer[prefix + "attn_norm"] = pin_tensor(torch.zeros(d_model, device=device, dtype=dtype_mapping[prefix + "attn_norm"]))
                if prefix == "o_m_" or not self.is_muon:
                    opt_layer[prefix + "q"] = pin_tensor(torch.zeros(d_model, n_heads * head_dim, device=device, dtype=dtype_mapping[prefix + "q"]))
                    opt_layer[prefix + "k"] = pin_tensor(torch.zeros(d_model, n_kv_heads * head_dim, device=device, dtype=dtype_mapping[prefix + "k"]))
                    opt_layer[prefix + "v"] = pin_tensor(torch.zeros(d_model, n_kv_heads * head_dim, device=device, dtype=dtype_mapping[prefix + "v"]))
                    opt_layer[prefix + "o"] = pin_tensor(torch.zeros(n_heads * head_dim, d_model, device=device, dtype=dtype_mapping[prefix + "o"]))
                opt_layer[prefix + "ffn_norm"] = pin_tensor(torch.zeros(d_model, device=device, dtype=dtype_mapping[prefix + "ffn_norm"]))
                if prefix == "o_m_" or not self.is_muon:
                    opt_layer[prefix + "1"] = pin_tensor(torch.zeros(d_model, expert_dim, device=device, dtype=dtype_mapping[prefix + "1"]))
                    opt_layer[prefix + "3"] = pin_tensor(torch.zeros(d_model, expert_dim, device=device, dtype=dtype_mapping[prefix + "3"]))
                    opt_layer[prefix + "2"] = pin_tensor(torch.zeros(expert_dim, d_model, device=device, dtype=dtype_mapping[prefix + "2"]))
        else:
            cur_offset = 0
            for prefix in prefixes:
                opt_layer[prefix + "attn_norm"] = buffer[cur_offset:cur_offset + d_model * dtype_mapping[prefix + "attn_norm"].itemsize].view(dtype_mapping[prefix + "attn_norm"]).reshape(d_model)
                cur_offset += d_model * dtype_mapping[prefix + "attn_norm"].itemsize
                if prefix == "o_m_" or not self.is_muon:
                    opt_layer[prefix + "q"] = buffer[cur_offset:cur_offset + d_model * n_heads * head_dim * dtype_mapping[prefix + "q"].itemsize].view(dtype_mapping[prefix + "q"]).reshape(d_model, n_heads * head_dim)
                    cur_offset += d_model * n_heads * head_dim * dtype_mapping[prefix + "q"].itemsize
                    opt_layer[prefix + "k"] = buffer[cur_offset:cur_offset + d_model * n_kv_heads * head_dim * dtype_mapping[prefix + "k"].itemsize].view(dtype_mapping[prefix + "k"]).reshape(d_model, n_kv_heads * head_dim)
                    cur_offset += d_model * n_kv_heads * head_dim * dtype_mapping[prefix + "k"].itemsize
                    opt_layer[prefix + "v"] = buffer[cur_offset:cur_offset + d_model * n_kv_heads * head_dim * dtype_mapping[prefix + "v"].itemsize].view(dtype_mapping[prefix + "v"]).reshape(d_model, n_kv_heads * head_dim)
                    cur_offset += d_model * n_kv_heads * head_dim * dtype_mapping[prefix + "v"].itemsize
                    opt_layer[prefix + "o"] = buffer[cur_offset:cur_offset + n_heads * head_dim * d_model * dtype_mapping[prefix + "o"].itemsize].view(dtype_mapping[prefix + "o"]).reshape(n_heads * head_dim, d_model)
                    cur_offset += n_heads * head_dim * d_model * dtype_mapping[prefix + "o"].itemsize
                opt_layer[prefix + "ffn_norm"] = buffer[cur_offset:cur_offset + d_model * dtype_mapping[prefix + "ffn_norm"].itemsize].view(dtype_mapping[prefix + "ffn_norm"]).reshape(d_model)
                cur_offset += d_model * dtype_mapping[prefix + "ffn_norm"].itemsize
                if prefix == "o_m_" or not self.is_muon:
                    opt_layer[prefix + "1"] = buffer[cur_offset:cur_offset + d_model * expert_dim * dtype_mapping[prefix + "1"].itemsize].view(dtype_mapping[prefix + "1"]).reshape(d_model, expert_dim)
                    cur_offset += d_model * expert_dim * dtype_mapping[prefix + "1"].itemsize
                    opt_layer[prefix + "3"] = buffer[cur_offset:cur_offset + d_model * expert_dim * dtype_mapping[prefix + "3"].itemsize].view(dtype_mapping[prefix + "3"]).reshape(d_model, expert_dim)
                    cur_offset += d_model * expert_dim * dtype_mapping[prefix + "3"].itemsize
                    opt_layer[prefix + "2"] = buffer[cur_offset:cur_offset + expert_dim * d_model * dtype_mapping[prefix + "2"].itemsize].view(dtype_mapping[prefix + "2"]).reshape(expert_dim, d_model)
                    cur_offset += expert_dim * d_model * dtype_mapping[prefix + "2"].itemsize
        
        return opt_layer, cur_offset
    
    def init_weights(self, weights):
        
        ## Using Pytorch default which is normal with 1 / sqrt(fan in)
        # 1. Norms → identity
        weights["w_attn_norm"].fill_(1.0)
        weights["w_ffn_norm"].fill_(1.0)

        # 2. Base stds
        model_std = 1 / np.sqrt(self.model_dims["d_model"])
        attn_dim = self.model_dims["n_heads"] * self.model_dims["head_dim"]
        attn_out_std = 1 / np.sqrt(attn_dim)
        expert_std = 1 / np.sqrt(self.model_dims["expert_dim"])

        # 3. Residual scaling factor
        resid_scale = 1 / np.sqrt(2 * self.model_dims["n_layers"])
        #resid_scale = 1.0

        # 4. Input projections (no residual scaling)
        weights["w_q"].normal_(mean=0.0, std=model_std)
        weights["w_k"].normal_(mean=0.0, std=model_std)
        weights["w_v"].normal_(mean=0.0, std=model_std)
        weights["w_1"].normal_(mean=0.0, std=model_std)
        weights["w_3"].normal_(mean=0.0, std=model_std)

        # 5. Output projections (WITH residual scaling)
        weights["w_o"].normal_(mean=0.0, std=attn_out_std * resid_scale)
        weights["w_2"].normal_(mean=0.0, std=expert_std * resid_scale)

        return weights
    
    def load(self, model_path, buffer = None, device = "cpu", pin_memory = True, is_opt=False, is_grad=False, dtype_mapping = None):

        if is_opt:
            return self.load_opt(model_path, buffer)

        if device != "cpu":
            pin_memory = False
        
        new_layer = self.create(buffer = buffer, device = device, pin_memory = pin_memory, is_grad=is_grad, dtype_mapping = dtype_mapping)
        
        if model_path is None:
            if is_grad:
                return new_layer
            else:
                return self.init_weights(new_layer)
        
        weight_names = [name for name in new_layer.keys()]

        for name in weight_names:
            weight_torch = torch.load(model_path + f"/layers/{self.layer_id}/{name}.pt", map_location="cpu")
            new_layer[name].copy_(weight_torch, non_blocking=True)
            del weight_torch

        return new_layer
    
    def load_opt(self, model_path, blank_opt):

        layer_id = self.layer_id
        for name, tensor in blank_opt.items():
            weight_torch = torch.load(model_path + f"/layers/{self.layer_id}/{name}.pt")
            tensor.copy_(weight_torch)

        return


    def save(self, weights, model_path, is_grad=False, is_opt=False):
        
        """
        ## weights are either params grads or opt state, only 1 of three
        prefixes = ["w_"]
        if is_grad:
            prefixes = ["g_"]

        if is_opt:
            prefixes = ["o_m_", "o_v_"]
        
        for prefix in prefixes:

            torch.save(weights[prefix + "attn_norm"], model_path + f"/layers/{self.layer_id}/{prefix}attn_norm.pt")
            torch.save(weights[prefix + "q"], model_path + f"/layers/{self.layer_id}/{prefix}q.pt")
            torch.save(weights[prefix + "k"], model_path + f"/layers/{self.layer_id}/{prefix}k.pt")
            torch.save(weights[prefix + "v"], model_path + f"/layers/{self.layer_id}/{prefix}v.pt")
            torch.save(weights[prefix + "o"], model_path + f"/layers/{self.layer_id}/{prefix}o.pt")
            torch.save(weights[prefix + "ffn_norm"], model_path + f"/layers/{self.layer_id}/{prefix}ffn_norm.pt")
            torch.save(weights[prefix + "1"], model_path + f"/layers/{self.layer_id}/{prefix}1.pt")
            torch.save(weights[prefix + "3"], model_path + f"/layers/{self.layer_id}/{prefix}3.pt")
            torch.save(weights[prefix + "2"], model_path + f"/layers/{self.layer_id}/{prefix}2.pt")
        """
        layer_id = self.layer_id
        for name, tensor in weights.items():
            torch.save(tensor, model_path + f"/layers/{layer_id}/{name}.pt")
        
    
    def make_chunk_metadata(self, seq_lens, seq_positions, prior_seq_lens, prior_seq_offsets, device, local_layer_ids=None):

        num_seqs = len(seq_lens)
        num_prior_seqs = len(prior_seq_lens)

        assert num_prior_seqs == num_seqs, "num_prior_seqs must be equal to num_seqs"

        total_q = sum(seq_lens)
        total_k = sum(prior_seq_lens) + total_q

        q_seq_offsets = torch.tensor([0] + list(np.cumsum(seq_lens)), dtype=torch.int32, device=device)
        k_seq_offsets = torch.tensor([0] + list(np.cumsum(np.array(seq_lens) + np.array(prior_seq_lens))), dtype=torch.int32, device=device)
        
        max_seqlen_q = max(seq_lens)
        max_seqlen_k = max([prior_seq_lens[i] + seq_lens[i] for i in range(num_seqs)])

        q_seq_lens = torch.tensor([seq_lens[i] for i in range(num_seqs)], dtype=torch.int32, device=device)
        k_seq_lens = torch.tensor([prior_seq_lens[i] + seq_lens[i] for i in range(num_seqs)], dtype=torch.int32, device=device)

        seq_positions = torch.tensor(seq_positions, dtype=torch.int32, device=device).reshape(-1, 1)

        chunk_metadata = {
            "seq_lens_host": seq_lens.copy(),
            "prior_seq_lens_host": prior_seq_lens.copy(),
            "prior_seq_offsets_host": prior_seq_offsets.copy(),
            "total_q": total_q,
            "total_k": total_k,
            "seq_positions": seq_positions,
            "q_seq_offsets": q_seq_offsets,
            "k_seq_offsets": k_seq_offsets,
            "q_seq_lens": q_seq_lens,
            "k_seq_lens": k_seq_lens,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k
        }

        return chunk_metadata

    def make_act_slot(self, num_tokens, saved_level, buffer=None, device="cpu", pin_memory=True):

        if saved_level is None:
            saved_level = self.max_saved_activations_level

        if device != "cpu":
            pin_memory = False

        d_model = self.model_dims["d_model"]
        n_heads = self.model_dims["n_heads"]
        n_kv_heads = self.model_dims["n_kv_heads"]
        head_dim = self.model_dims["head_dim"]
        expert_dim = self.model_dims["expert_dim"]

        resid_dtype = get_torch_dtype(self.model_dims["datatypes"]["residual"])

        ## assume activations are the same as residual
        attn_act_dtype = resid_dtype
        ffn_act_dtype = resid_dtype

        act_slot = {}
        if buffer is None:

        
            act_slot["attn_norm_rstd"] = pin_tensor(torch.zeros(num_tokens, 1, device=device, dtype=torch.float32))
            act_slot["ffn_norm_rstd"] = pin_tensor(torch.zeros(num_tokens, 1, device=device, dtype=torch.float32))
            act_slot["x_inp"] = pin_tensor(torch.zeros(num_tokens, d_model, device=device, dtype=torch.bfloat16))
            act_slot["xk"] = pin_tensor(torch.zeros(num_tokens, n_kv_heads, head_dim, device=device, dtype=attn_act_dtype))
            act_slot["xv"] = pin_tensor(torch.zeros(num_tokens, n_kv_heads, head_dim, device=device, dtype=attn_act_dtype))
            
            if saved_level >= 1:
                act_slot["attn_result"] = pin_tensor(torch.zeros(num_tokens, n_heads, head_dim, device=device, dtype=attn_act_dtype))
                act_slot["softmax_lse"] = pin_tensor(torch.zeros(n_heads, num_tokens, device=device, dtype=torch.float32))
                
            if saved_level >= 2:
                act_slot["xq"] = pin_tensor(torch.zeros(num_tokens, n_heads, head_dim, device=device, dtype=attn_act_dtype))
                act_slot["xo"] = pin_tensor(torch.zeros(num_tokens, d_model, device=device, dtype=attn_act_dtype))

            if saved_level >= 3:
                act_slot["x1"] = pin_tensor(torch.zeros(num_tokens, expert_dim, device=device, dtype=ffn_act_dtype))
                act_slot["x3"] = pin_tensor(torch.zeros(num_tokens, expert_dim, device=device, dtype=ffn_act_dtype))


        else:

            cur_offset = 0

            attn_norm_rtsd_size = num_tokens * torch.float32.itemsize
            ffn_norm_rtsd_size = num_tokens * torch.float32.itemsize
            x_inp_size = num_tokens * d_model * attn_act_dtype.itemsize
            xk_size = num_tokens * n_kv_heads * head_dim * attn_act_dtype.itemsize
            xv_size = num_tokens * n_kv_heads * head_dim * attn_act_dtype.itemsize
            attn_result_size = num_tokens * n_heads * head_dim * attn_act_dtype.itemsize
            softmax_lse_size = n_heads * num_tokens * torch.float32.itemsize
            xq_size = num_tokens * n_heads * head_dim * attn_act_dtype.itemsize
            xo_size = num_tokens * d_model * attn_act_dtype.itemsize
            x1_size = num_tokens * expert_dim * ffn_act_dtype.itemsize
            x3_size = num_tokens * expert_dim * ffn_act_dtype.itemsize

            act_slot["attn_norm_rstd"] = buffer[:attn_norm_rtsd_size].view(torch.float32).reshape(num_tokens, 1)
            cur_offset += attn_norm_rtsd_size
            act_slot["ffn_norm_rstd"] = buffer[cur_offset:cur_offset + ffn_norm_rtsd_size].view(torch.float32).reshape(num_tokens, 1)
            cur_offset += ffn_norm_rtsd_size
            act_slot["x_inp"] = buffer[cur_offset:cur_offset + x_inp_size].view(attn_act_dtype).reshape(num_tokens, d_model)
            cur_offset += x_inp_size
            act_slot["xk"] = buffer[cur_offset:cur_offset + xk_size].view(attn_act_dtype).reshape(num_tokens, n_kv_heads, head_dim)
            cur_offset += xk_size
            act_slot["xv"] = buffer[cur_offset:cur_offset + xv_size].view(attn_act_dtype).reshape(num_tokens, n_kv_heads, head_dim)
            cur_offset += xv_size

            if saved_level >= 1:
                act_slot["attn_result"] = buffer[cur_offset:cur_offset + attn_result_size].view(attn_act_dtype).reshape(num_tokens, n_heads, head_dim)
                cur_offset += attn_result_size
                act_slot["softmax_lse"] = buffer[cur_offset:cur_offset + softmax_lse_size].view(torch.float32).reshape(n_heads, num_tokens)
                cur_offset += softmax_lse_size
            if saved_level >= 2:
                act_slot["xq"] = buffer[cur_offset:cur_offset + xq_size].view(attn_act_dtype).reshape(num_tokens, n_heads, head_dim)
                cur_offset += xq_size
                act_slot["xo"] = buffer[cur_offset:cur_offset + xo_size].view(attn_act_dtype).reshape(num_tokens, d_model)
                cur_offset += xo_size
            if saved_level >= 3:
                act_slot["x1"] = buffer[cur_offset:cur_offset + x1_size].view(ffn_act_dtype).reshape(num_tokens, expert_dim)
                cur_offset += x1_size
                act_slot["x3"] = buffer[cur_offset:cur_offset + x3_size].view(ffn_act_dtype).reshape(num_tokens, expert_dim)
                cur_offset += x3_size

        total_size = 0
        for k, v in act_slot.items():
            total_size += v.numel() * v.dtype.itemsize

        return act_slot, total_size

    def get_act_slot_size(self, num_tokens, saved_level=None):

        total_size = 0

        if saved_level is None:
            saved_level = self.max_saved_activations_level

        # 1. Setup Dims
        d_model = self.model_dims["d_model"]
        n_heads = self.model_dims["n_heads"]
        n_kv_heads = self.model_dims["n_kv_heads"]
        head_dim = self.model_dims["head_dim"]
        # In dense layers, 'expert_dim' usually refers to the FFN intermediate size
        expert_dim = self.model_dims["expert_dim"] 

        # 2. Setup Dtypes
        # We mirror the logic in make_act_slot to ensure sizes match exactly
        resid_dtype = get_torch_dtype(self.model_dims["datatypes"]["residual"])
        
        # In your make_act_slot, these are set to resid_dtype
        attn_act_dtype = resid_dtype
        ffn_act_dtype = resid_dtype

        # ------------------------------------------------------------------
        # Level 0: Base Activations (Always Saved)
        # ------------------------------------------------------------------
        
        # Note: make_act_slot uses hardcoded torch.float32 for norms
        attn_norm_rstd_size = num_tokens * torch.float32.itemsize
        ffn_norm_rstd_size = num_tokens * torch.float32.itemsize
        
        # Note: make_act_slot uses hardcoded torch.bfloat16 for x_inp
        x_inp_size = num_tokens * d_model * torch.bfloat16.itemsize
        
        xk_size = num_tokens * n_kv_heads * head_dim * attn_act_dtype.itemsize
        xv_size = num_tokens * n_kv_heads * head_dim * attn_act_dtype.itemsize

        total_size += (attn_norm_rstd_size + ffn_norm_rstd_size + 
                       x_inp_size + xk_size + xv_size)

        if saved_level == 0:
            return total_size

        # ------------------------------------------------------------------
        # Level 1: Attention Results
        # ------------------------------------------------------------------
        
        attn_result_size = num_tokens * n_heads * head_dim * attn_act_dtype.itemsize
        # make_act_slot uses hardcoded torch.float32 for softmax_lse
        softmax_lse_size = n_heads * num_tokens * torch.float32.itemsize

        total_size += attn_result_size + softmax_lse_size

        if saved_level == 1:
            return total_size

        # ------------------------------------------------------------------
        # Level 2: Attention Projections (Q and Output)
        # ------------------------------------------------------------------
        
        xq_size = num_tokens * n_heads * head_dim * attn_act_dtype.itemsize
        xo_size = num_tokens * d_model * attn_act_dtype.itemsize

        total_size += xq_size + xo_size

        if saved_level == 2:
            return total_size

        # ------------------------------------------------------------------
        # Level 3: FFN Internals
        # ------------------------------------------------------------------
        
        # x1 and x3 are usually the Gate and Up projections in a SwiGLU FFN
        x1_size = num_tokens * expert_dim * ffn_act_dtype.itemsize
        x3_size = num_tokens * expert_dim * ffn_act_dtype.itemsize

        total_size += x1_size + x3_size

        # Level 3 is generally the max, return total
        return total_size

    def send_activations_home(self, home_act_slot, computed_act_slot, save_activations_level):

        save_level_mapping = {
            # START OF MIN SAVE (SAVED_LEVEL >= 0)
            "attn_norm_rstd": 0,
            "ffn_norm_rstd": 0,
            "x_inp": 0,
            "xk": 0,
            "xv": 0,
            ## START OF PARTIAL SAVE (SAVED_LEVEL >= 1)
            "attn_result": 1,
            "softmax_lse": 1,
            ## START OF FULL SAVE (SAVED_LEVEL >= 2)
            "xq": 2,
            "xo": 2,
            ## START OF MAX SAVE (SAVED_LEVEL >= 3)
            "x1": 3,
            "x3": 3,
        }

        for k, v in computed_act_slot.items():

            if k not in save_level_mapping:
                continue

            save_level = save_level_mapping[k]

            if save_level < 0:
                continue

            if save_level <= save_activations_level:
                home_act_slot[k].copy_(v, non_blocking=True)

    
    def fetch_activations(self, base_act_slot, home_act_slot, chunk_metadata, layer_id):

        act_slot = {}
        num_tokens = home_act_slot["x_inp"].shape[0]

        ## use view of base act slot with the correct shape for this chunk
        for k, v in base_act_slot.items():
            if k not in home_act_slot:
                continue
            if k != "softmax_lse" and v.shape[0] != num_tokens:
                act_slot[k] = v[:num_tokens, :]
            elif k == "softmax_lse" and v.shape[1] != num_tokens:
                act_slot[k] = v[:, :num_tokens]
            else:
                act_slot[k] = v
        
        for k, v in home_act_slot.items():
            act_slot[k].copy_(v, non_blocking=True)

        return act_slot

        
    def fetch_weights(self, gpu_weights, cpu_weights):
        for k, v in gpu_weights.items():
            gpu_weights[k].copy_(cpu_weights[k], non_blocking=True)
        
    def get_fwd_flops(self, chunk_metadata):
        # Initialize the saved dictionary for all levels (0 to 3)
        saved_fwd_flops = {}
        for i in range(self.max_saved_activations_level + 1):
            saved_fwd_flops[i] = 0

        num_tokens = chunk_metadata["total_q"]
        seq_lens = chunk_metadata["seq_lens_host"]
        prior_seq_lens = chunk_metadata["prior_seq_lens_host"]
        
        num_heads = self.model_dims["n_heads"]
        head_dim = self.model_dims["head_dim"]
        d_model = self.model_dims["d_model"]
        expert_dim = self.model_dims["expert_dim"] # FFN intermediate dimension
        n_kv_heads = self.model_dims["n_kv_heads"]

        # Effective dimension for attention calculations (all heads combined)
        attn_dim = num_heads * head_dim
        # Effective dimension for KV calculations
        kv_dim = n_kv_heads * head_dim

        fwd_flops = 0

        saved_fwd_flops[0] = 0

        for i in range(len(seq_lens)):
            seq_len = seq_lens[i]
            prior_seq_len = prior_seq_lens[i]

            # ----------------------------------------------------------------------
            # 1. KV Projections (xk, xv)
            # Mapping Level 0: "xk", "xv"
            # ----------------------------------------------------------------------
            # 2 (mul/add) * seq * d_model * kv_dim * 2 (for K and V)
            kv_proj_flops = 2 * seq_len * d_model * kv_dim * 2
            
            fwd_flops += kv_proj_flops

            # ----------------------------------------------------------------------
            # 2. Q and O Projections (xq, xo)
            # Mapping Level 2: "xq", "xo"
            # ----------------------------------------------------------------------
            # Q Proj: 2 * seq * d_model * attn_dim
            # O Proj: 2 * seq * d_model * attn_dim
            qo_flops = 2 * (2 * seq_len * d_model * attn_dim)
            
            fwd_flops += qo_flops
            
            # Saved at Level 2 and above
            saved_fwd_flops[2] += qo_flops
            saved_fwd_flops[3] += qo_flops

            # ----------------------------------------------------------------------
            # 3. Attention Mechanism (attn_result)
            # Mapping Level 1: "attn_result"
            # ----------------------------------------------------------------------
            # Cross attention to prior tokens (Full Causal)
            # 4 (2 for QK^T + 2 for AV) * seq * prior_seq * attn_dim
            attn_flops_prior = 4 * seq_len * prior_seq_len * attn_dim

            # Self attention (Current sequence)
            if self.model_dims["is_causal"]:
                # Causal divides operations by approx 2
                attn_flops_current = 2 * seq_len * seq_len * attn_dim
            else:
                # Full bidirectional
                attn_flops_current = 4 * seq_len * seq_len * attn_dim

            attn_flops = attn_flops_prior + attn_flops_current
            
            fwd_flops += attn_flops
            
            # Saved at Level 1 and above
            saved_fwd_flops[1] += attn_flops
            saved_fwd_flops[2] += attn_flops
            saved_fwd_flops[3] += attn_flops

            # ----------------------------------------------------------------------
            # 4. FFN Weights (x1, x3)
            # Mapping Level 3: "x1", "x3" (Gate and Up projections)
            # ----------------------------------------------------------------------
            # Assumes SwiGLU: x1=Gate, x3=Up. 
            # 2 (mul/add) * seq * d_model * expert_dim * 2 (matrices)
            ffn_up_gate_flops = 2 * seq_len * d_model * expert_dim * 2
            
            fwd_flops += ffn_up_gate_flops
            
            # Saved only at Max Level (3)
            saved_fwd_flops[3] += ffn_up_gate_flops

            # ----------------------------------------------------------------------
            # 5. FFN Down Projection
            # Not in mapping, assumed always recomputed or minimal save
            # ----------------------------------------------------------------------
            # 2 (mul/add) * seq * expert_dim * d_model
            ffn_down_flops = 2 * seq_len * expert_dim * d_model
            
            fwd_flops += ffn_down_flops

        return fwd_flops, saved_fwd_flops