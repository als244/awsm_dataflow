import torch
import numpy as np

import os

from .ops import awsm_rmsnorm_fwd, awsm_rmsnorm_fwd_recompute, awsm_rmsnorm_bwd, awsm_rope_fwd, awsm_rope_bwd, awsm_attention_fwd, awsm_softmax, awsm_attention_bwd, awsm_rmsnorm_bwd, awsm_adamw_step, awsm_muon_step
from .ops import awsm_moe_sort, awsm_moe_scatter, awsm_moe_scatter_routing_weights, awsm_moe_gather, awsm_copy_expert_counts, awsm_swiglu_moe_fwd, awsm_swiglu_moe_bwd, awsm_moe_router_gate_bwd, awsm_load_balance_bwd, awsm_fused_topk_softmax

from .matmul_dispatchers import dispatcher, dispatcher_secondary
from .mem_register import pin_tensor 
from .utils import *

class TransformerMoELayer():

    def __init__(self, layer_id, model_dims, model_hyperparams, is_muon=False, secondary_compute_stream=None):
        self.layer_id = layer_id
        self.model_dims = model_dims
        self.total_layers = self.model_dims["n_layers"]
        self.model_hyperparams = model_hyperparams
        self.step_num = 0
        self.is_muon = is_muon
        self.secondary_compute_stream = secondary_compute_stream
        self.expert_hist = torch.zeros(self.model_dims["num_routed_experts"], dtype=torch.int64, device="cpu")
        self.max_saved_activations_level = 3
    
    def forward(self, X, chunk_metadata, weights, base_act_slot, fwd_context):

        num_tokens = X.shape[0]

        top_k = self.model_dims["top_k"]

        act_slot = {}

        ## use view of base act slot with the correct shape for this chunk

        ## TODO: clean this up and have systematic way of handling act slots!!!!
        for k, v in base_act_slot.items():
            if k == "x_up":
                continue
            elif k == "expert_counts":
                act_slot[k] = v
            ## this is essentially free to recompute no need to make code have special cases...
            elif k == "scattered_router_weights":
                act_slot[k] = v[:num_tokens * top_k, :]
            elif k != "softmax_lse" and v.shape[0] != num_tokens:
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
        model_dim = self.model_dims["d_model"]

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

        ## this copy is a bit wasteful ,we could've had K and V matmul output directly go here...
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

        del x_temp

        # MLP part``
        ## the input is fwd_act_slot["xo"], so critically we don't overwrite it, but rather add this to output of moe and replace the original X which is transition!
        ## tricky bug that took a while to debug...!
        layer_output = self.forward_moe(attn_output_with_residual.view(-1, model_dim), chunk_metadata, weights, base_act_slot, act_slot, output_X=X)

        return layer_output, act_slot
    

    def forward_moe(self, input_X, chunk_metadata, weights, base_act_slot, act_slot, output_X=None):
        
        num_tokens = input_X.shape[0]

        ffn_norm_output, ffn_norm_rstd = awsm_rmsnorm_fwd(input_X, W=weights["w_ffn_norm"], rstd=act_slot["ffn_norm_rstd"], rms_norm_eps=self.model_hyperparams["rms_norm_eps"])    

        top_k = self.model_dims["top_k"]
        num_routed_experts = self.model_dims["num_routed_experts"]
        expert_dim = self.model_dims["expert_dim"]

        # ============================================================================
        # 1. Routing & Sorting
        # ============================================================================
        gate_logits = torch.matmul(ffn_norm_output, weights["w_router"], out=act_slot["x_router"])
        # raw_weights, topk_ids = torch.topk(gate_logits, k=top_k, dim=-1)
       
        # #`router_weights = torch.softmax(raw_weights, dim=-1)`
        # router_weights, _, _ = awsm_softmax(raw_weights, out=act_slot["router_weights"])
       
       
       
       
        # ## TODO: clean this up with small metadata and stay systematic. 
        # # Right now it is a bit fragile and "hardcoded"
        # #act_slot["router_weights"].copy_(router_weights)
        # topk_ids = topk_ids.int()
        # act_slot["chosen_experts"].copy_(topk_ids)

        router_weights, topk_ids = awsm_fused_topk_softmax(
            gate_logits, 
            top_k=top_k,
            topk_ids_out=act_slot["chosen_experts"],      # writes int32 directly, no cast/copy
            topk_weights_out=act_slot["router_weights"],   # writes softmax probs directly
        )

        # Get sort indices and the histogram of how many tokens each expert gets

        index_mapping = chunk_metadata["token_index_mapping"][self.layer_id]
        expert_counts_gpu = act_slot["expert_counts"]
        indices, expert_counts_gpu = awsm_moe_sort(topk_ids, num_experts=num_routed_experts, indices=index_mapping, expert_counts_gpu=expert_counts_gpu)

        # ============================================================================
        # 2. Scattering (Tokens & Weights)
        # ============================================================================
        scattered_x = torch.empty((num_tokens * top_k, self.model_dims["d_model"]), dtype=input_X.dtype, device=input_X.device)
        scattered_router_weights = act_slot["scattered_router_weights"]

        # A. Scatter Input Tokens [T, D] -> [T*K, D]
        x_sorted = awsm_moe_scatter(ffn_norm_output, indices, out=scattered_x)
        
        ## now done with ffn norm output
        del ffn_norm_output

        # B. Scatter Router Weights [T, K] -> [T*K]
        router_weights_sorted = awsm_moe_scatter_routing_weights(router_weights, indices, out=scattered_router_weights)
        
        # ============================================================================
        # 3. Get Offsets on CPU for Correct Dispatching & Set Activation Partitions
        # ============================================================================
        expert_counts_cpu = chunk_metadata["expert_counts_host"][self.layer_id]
        awsm_copy_expert_counts(expert_counts_gpu, expert_counts_cpu)
        
        
        ## CRITICAL: need towait for copy to finish before dispatching`
        ## we did this on compute stream and had kernel write directly to memmapped
        ## host memory in order to bypass D->H PCIe stream`
        torch.cuda.current_stream().synchronize()
        self.expert_hist.add_(expert_counts_cpu)
        expert_counts_cumsum = torch.cumsum(expert_counts_cpu, dim=0)

        assert expert_counts_cpu.sum() == num_tokens * top_k

        max_expert_tokens = expert_counts_cpu.max()

        X_act_even = torch.zeros((max_expert_tokens, self.model_dims["expert_dim"]), dtype=input_X.dtype, device=input_X.device)
        X_act_odd = torch.zeros((max_expert_tokens, self.model_dims["expert_dim"]), dtype=input_X.dtype, device=input_X.device)
        
        
        # ============================================================================
        # 4. Expert Execution Loop
        # ============================================================================
        
        primary_stream = torch.cuda.current_stream()
        primary_dispatcher = dispatcher
        primary_stream_ptr = primary_stream.cuda_stream
        use_secondary = False
        if self.secondary_compute_stream is not None:
            secondary_stream = self.secondary_compute_stream
            secondary_stream_ptr = secondary_stream.cuda_stream
            secondary_dispatcher = dispatcher_secondary
            secondary_stream.wait_stream(primary_stream)
            use_secondary = True
        else:
            secondary_stream = primary_stream
            secondary_dispatcher = dispatcher
            secondary_stream_ptr = primary_stream.cuda_stream


        cur_offset = 0
        act_slot["x_up"] = {}

        for expert_id in range(num_routed_experts):
            
            num_exp_tokens = expert_counts_cpu[expert_id].item()

            if num_exp_tokens == 0:
                continue

            start = cur_offset
            end = cur_offset + num_exp_tokens
            
            act_slot["x_up"][expert_id] = base_act_slot["x_up"][start:end, :]
            cur_offset += num_exp_tokens

            if expert_id % 2 == 0:
                cur_dispatcher = primary_dispatcher
                cur_stream_ptr = primary_stream_ptr
                cur_stream = primary_stream
                ## temp memory
                X_act = X_act_even[:num_exp_tokens, :]
            else:
                cur_dispatcher = secondary_dispatcher
                cur_stream_ptr = secondary_stream_ptr
                cur_stream = secondary_stream
                ## temp memory
                X_act = X_act_odd[:num_exp_tokens, :]

            with cur_stream:
                X_inp = x_sorted[start:end, :]
                weights_up = weights["w_up"][expert_id, :, :]

                ## x1 and x3 matmuls
                X_preact = act_slot["x_up"][expert_id]
                cur_dispatcher.matmul(cur_stream_ptr, A=X_inp, B=weights_up, D=X_preact)

                ## swiglu activation and scaling by router weights after activation
                #router_weights_inp = router_weights_sorted[start:end]

                ## assumes X_preact is (T, 2*F) and router_weights_inp is (T, K)
                ## splits each row into two parts, one for the gate and one for the value
                #X_act = awsm_swiglu_moe_fwd(X_preact, router_weights_inp, out=X_act)
                awsm_swiglu_moe_fwd(X_preact, out=X_act)

                weights_down = weights["w_down"][expert_id, :, :]

                ## can overwrite X_inp now...
                cur_dispatcher.matmul(cur_stream_ptr, A=X_act, B=weights_down, D=X_inp)
        
    
        if use_secondary:
            primary_stream.wait_stream(secondary_stream)

        # ## for python gc memory safety sync here before any deletes
        # primary_stream.synchronize()

        # ============================================================================
        # 5. Merge Expert Outputs
        # ============================================================================

        if output_X is None:
            output_X = torch.empty_like(input_X)

        with primary_stream:
            ## expert outputs were written into scattered_x, so we need to gather them back...
            ## overwrite output_X and add the residual (input to block pre norm)
            awsm_moe_gather(scattered_x, indices, residual=input_X, weights=router_weights, out=output_X)

        del X_act_even
        del X_act_odd
        del scattered_x

        # Return output
        return output_X

    def forward_moe_recompute(self, X, chunk_metadata, weights, base_act_slot, fwd_act_slot):

        num_tokens = X.shape[0]

        fwd_act_slot["ffn_norm_output"] = awsm_rmsnorm_fwd_recompute(X, weights["w_ffn_norm"], fwd_act_slot["ffn_norm_rstd"])
        ffn_norm_output = fwd_act_slot["ffn_norm_output"]
       

        top_k = self.model_dims["top_k"]
        num_experts = self.model_dims["num_routed_experts"]
        expert_dim = self.model_dims["expert_dim"]

        ## populated from forward
        router_weights = fwd_act_slot["router_weights"]
        index_mapping = chunk_metadata["token_index_mapping"][self.layer_id]
        expert_counts_cpu = chunk_metadata["expert_counts_host"][self.layer_id]

        assert expert_counts_cpu.sum() == num_tokens * top_k
    

        ## Re-Do Scatter
        scattered_x = torch.empty((num_tokens * top_k, self.model_dims["d_model"]), dtype=X.dtype, device=X.device)
        routed_weights_sorted = fwd_act_slot["scattered_router_weights"]

        # A. Scatter Input Tokens [T, D] -> [T*K, D]
        x_sorted = awsm_moe_scatter(ffn_norm_output, index_mapping, out=scattered_x)

        ## save for usage in backprop
        fwd_act_slot["scattered_x"] = scattered_x

        # B. Scatter Router Weights [T, K] -> [T*K]

        ## always storing this instead
        #router_weights_sorted = awsm_moe_scatter_routing_weights(router_weights, indices, out=scattered_router_weights)

        # ============================================================================
        # Do Expert Computation Loop, but only Up Proj
        # ============================================================================

        primary_stream = torch.cuda.current_stream()
        primary_dispatcher = dispatcher
        primary_stream_ptr = primary_stream.cuda_stream
        use_secondary = False
        if self.secondary_compute_stream is not None:
            secondary_stream = self.secondary_compute_stream
            secondary_stream_ptr = secondary_stream.cuda_stream
            secondary_dispatcher = dispatcher_secondary
            secondary_stream.wait_stream(primary_stream)
            use_secondary = True
        else:
            secondary_stream = primary_stream
            secondary_dispatcher = dispatcher
            secondary_stream_ptr = primary_stream.cuda_stream

        
        fwd_act_slot["x_up"] = {}
        cur_offset = 0
        
        for expert_id in range(num_experts):

            num_exp_tokens = expert_counts_cpu[expert_id].item()

            if num_exp_tokens == 0:
                continue
            
            start = cur_offset
            end = cur_offset + num_exp_tokens

            fwd_act_slot["x_up"][expert_id] = base_act_slot["x_up"][start:end, :]
            cur_offset += num_exp_tokens

            if expert_id % 2 == 0:
                cur_dispatcher = primary_dispatcher
                cur_stream_ptr = primary_stream_ptr
                cur_stream = primary_stream
            else:
                cur_dispatcher = secondary_dispatcher
                cur_stream_ptr = secondary_stream_ptr
                cur_stream = secondary_stream

            with cur_stream:
                X_inp = x_sorted[start:end, :]
                weights_up = weights["w_up"][expert_id, :, :]

                ## x1 and x3 matmuls
                cur_dispatcher.matmul(cur_stream_ptr, A=X_inp, B=weights_up, D=fwd_act_slot["x_up"][expert_id])
        

        if use_secondary:
            primary_stream.wait_stream(secondary_stream)

        # ## for python gc memory safety sync here before any deletes
        # primary_stream.synchronize()

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
        model_dim = self.model_dims["d_model"]

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

        
        if "x_up" not in fwd_act_slot:
            self.forward_moe_recompute(attn_output_with_residual.view(-1, model_dim), chunk_metadata, weights, base_act_slot, fwd_act_slot)
        
        return fwd_act_slot
    
    def backward_moe(self, dX, chunk_metadata, weights, grad_weights, fwd_act_slot, tokens_per_step, prior_ffn_norm_upstream=None):

        num_tokens = dX.shape[0]
        top_k = self.model_dims["top_k"]
        num_experts = self.model_dims["num_routed_experts"]
        model_dim = self.model_dims["d_model"]
        expert_dim = self.model_dims["expert_dim"]

        fwd_x_up = fwd_act_slot["x_up"]
        index_mapping = chunk_metadata["token_index_mapping"][self.layer_id]
        expert_counts_cpu = chunk_metadata["expert_counts_host"][self.layer_id]
        router_weights = fwd_act_slot["router_weights"]
        scattered_router_weights = fwd_act_slot["scattered_router_weights"]
        chosen_experts = fwd_act_slot["chosen_experts"]

        assert expert_counts_cpu.sum() == num_tokens * top_k

        max_expert_tokens = expert_counts_cpu.max()  

        ## 1.) do scatter of upstream gradient (undoing the gather from fwd)
        scattered_upstream = torch.zeros(num_tokens * top_k, model_dim, dtype=dX.dtype, device=dX.device)
    
        ## This is T*K x D matrix using the same index mapping as from the original forward scatter
        #scattered_upstream = awsm_moe_scatter(dX, index_mapping, scales=scattered_router_weights, out=scattered_upstream)
        scattered_upstream = awsm_moe_scatter(dX, index_mapping, out=scattered_upstream)
        ## 2.) Backprop through each expert

        ## will need ffn_norm_output later on anyways 
        # (could potentially get away with waiting to alloc memory to reduce peak usage slightly)
        if "ffn_norm_output" in fwd_act_slot:
            fwd_ffn_norm_output = fwd_act_slot.pop("ffn_norm_output")
        else:
            fwd_ffn_norm_output = awsm_rmsnorm_fwd_recompute(fwd_act_slot["xo"], weights["w_ffn_norm"], fwd_act_slot["ffn_norm_rstd"])

        ## will need the original scattered x for model gradients with respect to W_up
        ## check if we might already have it if we took the forward_moe_recompute path
        if "scattered_x" not in fwd_act_slot:
            scattered_x = torch.zeros(num_tokens * top_k, model_dim, dtype=dX.dtype, device=dX.device)
            scattered_x = awsm_moe_scatter(fwd_ffn_norm_output, index_mapping, out=scattered_x)
        else:
            scattered_x = fwd_act_slot.pop("scattered_x")

        ## along with backprop through datapathm each token within expert
        ## computes derivative with respect to its given router weight
        ## (dot product of upstream gradient of activation and result of forward activation)
        dprobs = torch.zeros_like(scattered_router_weights)

        ## temporary buffers for working space during each expert backpropr

        ## a.) exp_num_tokens * expert_dim (X_act_up)
        ## b.) exp_num_tokens * 2 * expert_dim (dx_up_up)
        ## c.) exp_num_tokens * expert_dim (fwd_act)
        ## d.) exp_num_tokens * expert_dim (ffn_norm_output)
        
        X_even_temp = torch.zeros(max_expert_tokens * (4 * expert_dim), dtype=dX.dtype, device=dX.device)    
        X_odd_temp = torch.zeros(max_expert_tokens * (4 * expert_dim), dtype=dX.dtype, device=dX.device)
        
        
        ### Prepare streams for dispatching
        primary_stream = torch.cuda.current_stream()
        primary_dispatcher = dispatcher
        primary_stream_ptr = primary_stream.cuda_stream
        use_secondary = False
        if self.secondary_compute_stream is not None:
            secondary_stream = self.secondary_compute_stream
            secondary_stream_ptr = secondary_stream.cuda_stream
            secondary_dispatcher = dispatcher_secondary
            secondary_stream.wait_stream(primary_stream)
            use_secondary = True
        else:
            secondary_stream = primary_stream
            secondary_dispatcher = dispatcher
            secondary_stream_ptr = primary_stream.cuda_stream

        ### Dispatch through each expert

        cur_offset = 0

        for expert_id in range(num_experts):

            
            num_exp_tokens = expert_counts_cpu[expert_id].item()

            if num_exp_tokens == 0:
                continue

            start = cur_offset
            end = cur_offset + num_exp_tokens
            cur_offset += num_exp_tokens
            
            if expert_id % 2 == 0:
                cur_dispatcher = primary_dispatcher
                cur_stream_ptr = primary_stream_ptr
                cur_stream = primary_stream
                X_temp = X_even_temp
            else:
                cur_dispatcher = secondary_dispatcher
                cur_stream_ptr = secondary_stream_ptr
                cur_stream = secondary_stream
                X_temp = X_odd_temp


            with cur_stream:
                ### a.) Backprop through W_down
                # exp_upstream is SCATTERED AND unscaled
                # This produces dx_act_up which is also unscaled
                exp_upstream = scattered_upstream[start:end, :]
                W_down = weights["w_down"][expert_id, :, :]

                temp_offset = 0

                dx_act_up = X_temp[temp_offset:temp_offset + num_exp_tokens * expert_dim].view(num_exp_tokens, expert_dim)
                temp_offset += num_exp_tokens * expert_dim

                cur_dispatcher.matmul(cur_stream_ptr, A=exp_upstream, B=W_down.T, D=dx_act_up)

                ### b.) SwiGLU Backward
                # - Input dx_act_up is unscaled.
                # - Kernel does dot product (rowwise) of dx_act_up and recomputed fwd act to get router gradient
                # - Kernel interally rescales dx_act_up in order to return correct dx_up_up
                exp_dprobs = dprobs[start:end]
                exp_probs = scattered_router_weights[start:end]
                X_preact = fwd_act_slot["x_up"][expert_id]

                dx_up_up = X_temp[temp_offset:temp_offset + num_exp_tokens * 2 * expert_dim].view(num_exp_tokens, 2 * expert_dim)
                temp_offset += num_exp_tokens * 2 * expert_dim
                fwd_act = X_temp[temp_offset:temp_offset + num_exp_tokens * expert_dim].view(num_exp_tokens, expert_dim)
    

                ## Could have also used variant where we scale before the upstream scatter...
                # dx_up_up, exp_dprobs = awsm_swiglu_moe_bwd_prescaled(
                #     dx_act_up, X_preact, exp_probs, 
                #     dx=dx_up_up, dw=exp_dprobs, fwd_act=fwd_act
                # )

                ## out fwd act is the scaled variant
                dx_up_up, exp_dprobs = awsm_swiglu_moe_bwd(
                    dx_act_up, X_preact, exp_probs, 
                    dx=dx_up_up, dw=exp_dprobs, fwd_act=fwd_act
                )

                ### c.) Compute gradients for W_down
                # We use exp_upstream (Unscaled) directly.
                # dW = Input Scaled.T @ Grad_Output_Unscaled

                G_down = grad_weights["g_down"][expert_id, :, :]
                cur_dispatcher.matmul(
                    cur_stream_ptr, 
                    A=fwd_act.T, B=exp_upstream, 
                    C=G_down, 
                    D=G_down, 
                    beta=1.0, alpha=1.0
                )

                ### d.) Backprop through W_up
                # dx_up_up is SCALED.
                # We overwrite exp_upstream with the result to be gathered later.
                # It now references DOWNSTREAM GRADIENTS to be gathered
                W_up = weights["w_up"][expert_id, :, :]

                cur_dispatcher.matmul(cur_stream_ptr, A=dx_up_up, B=W_up.T, D=exp_upstream)

                ### d.) compute model gradients for W_up for this expert
                ## - get original inputs from the scattered_x from fwd

                ## num_exp_tokens x model_dim
                exp_inp = scattered_x[start:end, :]

                G_up = grad_weights["g_up"][expert_id, :, :]

                cur_dispatcher.matmul(cur_stream_ptr, A=exp_inp.T, B=dx_up_up, C=G_up, D=G_up, beta=1.0, alpha=1.0)
        

        ## now at end of loop scattered_upstream contains the datapath upstream gradients of FFN norm
        ## we need to gather these and then add on upstream gradient from the router path

        if use_secondary:
            primary_stream.wait_stream(secondary_stream)
        
        # ## for python gc memory safety sync here before any deletes 
        # # (python not tracking objs/launches from dispatcher, so unclear about when objs are released)
        # primary_stream.synchronize()

        del scattered_x
        del X_even_temp
        del X_odd_temp
    
        ## gather the scattered upstream gradients

        if prior_ffn_norm_upstream is None:
            ffn_norm_upstream = torch.zeros_like(dX)
        else:
            ffn_norm_upstream = prior_ffn_norm_upstream

        ffn_norm_upstream = awsm_moe_gather(scattered_upstream, index_mapping, out=ffn_norm_upstream)

        ## compute gradients from router path

        dlogits = torch.zeros(num_tokens, num_experts, dtype=dX.dtype, device=dX.device)

        ## OPTIONAL: load balance loss
        ## now placing this after to not impact data gradient flow...?
        # load_bal_coeff = self.model_hyperparams.get("load_bal_coeff", 0.0)
        # if load_bal_coeff > 0.0:
        #     dlogits = awsm_load_balance_bwd(
        #         logits=fwd_act_slot["x_router"],
        #         expert_counts=fwd_act_slot["expert_counts"],
        #         num_experts=num_experts,
        #         alpha=load_bal_coeff,
        #         top_k=top_k  # Add this parameter
        #     )
        # else:
        #     dlogits = torch.zeros(num_tokens, num_experts, dtype=dX.dtype, device=dX.device)
        

        dlogits = awsm_moe_router_gate_bwd(router_weights, dprobs, index_mapping, chosen_experts, dlogits=dlogits)


        ## ffn norm backwards

        ### Optional load balance loss, applying before we determine the gradient
        ### that should be accumulated into upstream of ffn (dlogits x w_router.T)

        ## We are not detaching this and having gradients flow to earlier layers
        ## to enforce better routability
        load_bal_coeff = self.model_hyperparams.get("load_bal_coeff", 0.0)
        if load_bal_coeff > 0.0:
            ## accumulates new gradients into existing dlogits
            dlogits = awsm_load_balance_bwd(
                logits=fwd_act_slot["x_router"],
                expert_counts=fwd_act_slot["expert_counts"],
                num_experts=num_experts,
                alpha=load_bal_coeff,
                tokens_per_step=tokens_per_step,
                top_k=top_k,
                dlogits=dlogits
            )

        ## compute backprop through router branch and model gradients of router
        dispatcher.matmul(primary_stream_ptr, A=dlogits, B=weights["w_router"].T, C=ffn_norm_upstream, D=ffn_norm_upstream, beta=1.0, alpha=1.0)

        ## Compute router gradients with load balacing loss tacked on
        dispatcher.matmul(primary_stream_ptr, A=fwd_ffn_norm_output.T, B=dlogits, C=grad_weights["g_router"], D=grad_weights["g_router"], beta=1.0, alpha=1.0)


        ## add on new gradients with block's upstream residual gradient

        ## took a LONG time to fix this bug where before we were passing in fwd_ffn_output instead of fwd_ffn_norm_input as 2nd arg...!!
        fwd_ffn_norm_input = fwd_act_slot["xo"]

        ffn_norm_downstream, _, _ = awsm_rmsnorm_bwd(ffn_norm_upstream, fwd_ffn_norm_input, weights["w_ffn_norm"], fwd_act_slot["ffn_norm_rstd"], dX=dX, dW=grad_weights["g_ffn_norm"])
        
        # ## for safety sync here before any deletes
        # primary_stream.synchronize()
        
        ## careful not to free
        del scattered_upstream
        del dlogits
        del dprobs

        return ffn_norm_downstream
        

    def backward(self, dX, chunk_metadata, weights, grad_weights, fwd_act_slot, fwd_context, bwd_context, total_tokens_per_step=None):

        ## 1st if shared experts can process each individually/seqeuentially
        ## and accumulating bwdX results in "ffn_norm_upstream" and can then 
        # pass this to backward moe for further accumulation for bwdX usage


        ### Part 1. MoE

        ### a.) Scatter Upstream Gradient
        ### b.) Backprop through Expert Weights
        ### c.) Gather Expert Gradients (Accumulate across all experts into upstream gradient of FFN norm)
        ### d.) !!! Need to think about this and make kernel for router gate gradient... Backprop through router gate to get router upstream gradient, then backprop through router to get router downstream gradient
        ### e.) Accumulate results of gathered graidents nad result of router downstream gradient into final upstream grad of FFN norm

        ## Accumulate gradients from MoE Block into same input matrix passed in
        dX = self.backward_moe(dX, chunk_metadata, weights, grad_weights, fwd_act_slot, total_tokens_per_step)

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
        ret = awsm_adamw_step(weights["w_router"], grad_weights["g_router"], opt_state["o_m_router"], opt_state["o_v_router"], 
               lr=lr, beta1=beta1, beta2=beta2, eps=eps, 
               weight_decay=weight_decay, step=step_num)
        if ret != 0:
            print(f"AdamW Step failed for router at layer {self.layer_id}")
            return -1
        ret = awsm_adamw_step(weights["w_up"], grad_weights["g_up"], opt_state["o_m_up"], opt_state["o_v_up"], 
               lr=lr, beta1=beta1, beta2=beta2, eps=eps, 
               weight_decay=weight_decay, step=step_num)
        if ret != 0:
            print(f"AdamW Step failed for up at layer {self.layer_id}")
            return -1
        ret = awsm_adamw_step(weights["w_down"], grad_weights["g_down"], opt_state["o_m_down"], opt_state["o_v_down"], 
               lr=lr, beta1=beta1, beta2=beta2, eps=eps, 
               weight_decay=weight_decay, step=step_num)
        if ret != 0:
            print(f"AdamW Step failed for down at layer {self.layer_id}")
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

        num_experts = self.model_dims["num_routed_experts"]

        ### Norms use AdamW

        ret = awsm_adamw_step(weights["w_attn_norm"], grad_weights["g_attn_norm"], opt_state["o_m_attn_norm"], opt_state["o_v_attn_norm"], 
               lr=lr, beta1=beta1, beta2=beta2, eps=eps, 
               weight_decay=weight_decay, step=step_num)
        if ret != 0 :
            print(f"AdamW Step failed for attn norm at layer {self.layer_id}")
            return -1
        
        ret = awsm_adamw_step(weights["w_ffn_norm"], grad_weights["g_ffn_norm"], opt_state["o_m_ffn_norm"], opt_state["o_v_ffn_norm"], 
               lr=lr, beta1=beta1, beta2=beta2, eps=eps, 
               weight_decay=weight_decay, step=step_num)
        if ret != 0:
            print(f"AdamW Step failed for ffn norm at layer {self.layer_id}")
            return -1

        ## Have router weights be AdamW
        ret = awsm_adamw_step(weights["w_router"], grad_weights["g_router"], opt_state["o_m_router"], opt_state["o_v_router"], 
               lr=lr, beta1=beta1, beta2=beta2, eps=eps, 
               weight_decay=weight_decay, step=step_num)
        if ret != 0:
            print(f"AdamW Step failed for router at layer {self.layer_id}")
            return -1
    
        ### Do Muon Steps
        ret = awsm_muon_step(weights["w_q"], grad_weights["g_q"], opt_state["o_m_q"], 
               lr=lr, beta=muon_beta, eps=eps,weight_decay=weight_decay)
        if ret != 0:
            print(f"Muon Step failed for q at layer {self.layer_id}")
            return -1
        ret = awsm_muon_step(weights["w_k"], grad_weights["g_k"], opt_state["o_m_k"], 
               lr=lr, beta=muon_beta, eps=eps,weight_decay=weight_decay)
        if ret != 0:
            print(f"Muon Step failed for k at layer {self.layer_id}")
            return -1
        ret = awsm_muon_step(weights["w_v"], grad_weights["g_v"], opt_state["o_m_v"], 
               lr=lr, beta=muon_beta, eps=eps,weight_decay=weight_decay)
        if ret != 0:
            print(f"Muon Step failed for v at layer {self.layer_id}")
            return -1
        ret = awsm_muon_step(weights["w_o"], grad_weights["g_o"], opt_state["o_m_o"], 
               lr=lr, beta=muon_beta, eps=eps,weight_decay=weight_decay)
        if ret != 0:
            print(f"Muon Step failed for o at layer {self.layer_id}")
            return -1

        expert_dim = self.model_dims["expert_dim"]
        for e in range(num_experts):

            ## w1
            ret = awsm_muon_step(weights["w_up"][e, :, :], grad_weights["g_up"][e, :, :], opt_state["o_m_up"][e, :, :], 
                    lr=lr, beta=muon_beta, eps=eps,weight_decay=weight_decay)
            if ret != 0:
                print(f"Muon Step failed for w_up {e} at layer {self.layer_id}")
                return -1
            ## w2
            ret = awsm_muon_step(weights["w_down"][e, :, :], grad_weights["g_down"][e, :, :], opt_state["o_m_down"][e, :, :], 
                    lr=lr, beta=muon_beta, eps=eps,weight_decay=weight_decay)
            if ret != 0:
                print(f"Muon Step failed for w_down {e} at layer {self.layer_id}")
                return -1

        return 0


         
    
    def create(self, buffer = None, device = "cpu", pin_memory = True, is_grad=False, dtype_mapping = None):

        d_model = self.model_dims["d_model"]
        n_heads = self.model_dims["n_heads"]
        n_kv_heads = self.model_dims["n_kv_heads"]
        head_dim = self.model_dims["head_dim"]
        expert_dim = self.model_dims["expert_dim"]
        num_experts = self.model_dims["num_routed_experts"]

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
                prefix + "router": torch.bfloat16,
                prefix + "up": torch.bfloat16,
                prefix + "down": torch.bfloat16,
            }

        if device != "cpu":
            pin_memory = False

        

        if buffer is None:

            ### avoid using pin_memory as causes excessive fragmentation from using cudaHostAlloc, better off to use cpu allocator and then register pointer...
            # return {
            #     f"{prefix}attn_norm": torch.zeros(d_model, device=device, dtype=dtype_mapping[prefix + "attn_norm"], pin_memory=pin_memory),
            #     f"{prefix}q": torch.zeros(d_model, n_heads * head_dim, device=device, dtype=dtype_mapping[prefix + "q"], pin_memory=pin_memory),
            #     f"{prefix}k": torch.zeros(d_model, n_kv_heads * head_dim, device=device, dtype=dtype_mapping[prefix + "k"], pin_memory=pin_memory),
            #     f"{prefix}v": torch.zeros(d_model, n_kv_heads * head_dim, device=device, dtype=dtype_mapping[prefix + "v"], pin_memory=pin_memory),
            #     f"{prefix}o": torch.zeros(n_heads * head_dim, d_model, device=device, dtype=dtype_mapping[prefix + "o"], pin_memory=pin_memory),
            #     f"{prefix}ffn_norm": torch.zeros(d_model, device=device, dtype=dtype_mapping[prefix + "ffn_norm"], pin_memory=pin_memory),
            #     f"{prefix}router": torch.zeros(d_model, num_experts, device=device, dtype=dtype_mapping[prefix + "router"], pin_memory=pin_memory),
            #     f"{prefix}up": torch.zeros(num_experts, d_model, 2 * expert_dim, device=device, dtype=dtype_mapping[prefix + "up"], pin_memory=pin_memory),
            #     f"{prefix}down": torch.zeros(num_experts, expert_dim, d_model, device=device, dtype=dtype_mapping[prefix + "down"], pin_memory=pin_memory),
            # }
            return {
                f"{prefix}attn_norm": pin_tensor(torch.zeros(d_model, device=device, dtype=dtype_mapping[prefix + "attn_norm"])),
                f"{prefix}q": pin_tensor(torch.zeros(d_model, n_heads * head_dim, device=device, dtype=dtype_mapping[prefix + "q"])),
                f"{prefix}k": pin_tensor(torch.zeros(d_model, n_kv_heads * head_dim, device=device, dtype=dtype_mapping[prefix + "k"])),
                f"{prefix}v": pin_tensor(torch.zeros(d_model, n_kv_heads * head_dim, device=device, dtype=dtype_mapping[prefix + "v"])),
                f"{prefix}o": pin_tensor(torch.zeros(n_heads * head_dim, d_model, device=device, dtype=dtype_mapping[prefix + "o"])),
                f"{prefix}ffn_norm": pin_tensor(torch.zeros(d_model, device=device, dtype=dtype_mapping[prefix + "ffn_norm"])),
                f"{prefix}router": pin_tensor(torch.zeros(d_model, num_experts, device=device, dtype=dtype_mapping[prefix + "router"])),
                f"{prefix}up": pin_tensor(torch.zeros(num_experts, d_model, 2 * expert_dim, device=device, dtype=dtype_mapping[prefix + "up"])),
                f"{prefix}down": pin_tensor(torch.zeros(num_experts, expert_dim, d_model, device=device, dtype=dtype_mapping[prefix + "down"])),
            }
        else:
            layer = {}
            attn_norm_size = d_model * dtype_mapping[prefix + "attn_norm"].itemsize
            q_size = d_model * n_heads * head_dim * dtype_mapping[prefix + "q"].itemsize
            k_size = d_model * n_kv_heads * head_dim * dtype_mapping[prefix + "k"].itemsize
            v_size = d_model * n_kv_heads * head_dim * dtype_mapping[prefix + "v"].itemsize
            o_size = n_heads * head_dim * d_model * dtype_mapping[prefix + "o"].itemsize
            ffn_norm_size = d_model * dtype_mapping[prefix + "ffn_norm"].itemsize
            router_size = d_model * num_experts * dtype_mapping[prefix + "router"].itemsize
            x_up_size = num_experts * d_model * 2 * expert_dim * dtype_mapping[prefix + "up"].itemsize
            x_down_size = num_experts * expert_dim * d_model * dtype_mapping[prefix + "down"].itemsize

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
            layer[prefix + "router"] = buffer[cur_offset:cur_offset + router_size].view(dtype_mapping[prefix + "router"]).reshape(d_model, num_experts)
            cur_offset += router_size
            layer[prefix + "up"] = buffer[cur_offset:cur_offset + x_up_size].view(dtype_mapping[prefix + "up"]).reshape(num_experts, d_model, 2 * expert_dim)
            cur_offset += x_up_size
            layer[prefix + "down"] = buffer[cur_offset:cur_offset + x_down_size].view(dtype_mapping[prefix + "down"]).reshape(num_experts, expert_dim, d_model)
            cur_offset += x_down_size

            return layer

    def create_opt(self, buffer = None, device = "cpu", pin_memory = True, dtype_mapping = None):

        is_muon = self.is_muon

        d_model = self.model_dims["d_model"]
        n_heads = self.model_dims["n_heads"]
        n_kv_heads = self.model_dims["n_kv_heads"]
        head_dim = self.model_dims["head_dim"]
        expert_dim = self.model_dims["expert_dim"]
        num_experts = self.model_dims["num_routed_experts"]

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
                dtype_mapping[prefix + "router"] = torch.bfloat16
                dtype_mapping[prefix + "up"] = torch.bfloat16
                dtype_mapping[prefix + "down"] = torch.bfloat16

        opt_layer = {}     
        cur_offset = 0
        if buffer is None:

            for prefix in prefixes:
                opt_layer[prefix + "attn_norm"] = pin_tensor(torch.zeros(d_model, device=device, dtype=dtype_mapping[prefix + "attn_norm"]))
                if prefix == "o_m_" or not is_muon:
                    opt_layer[prefix + "q"] = pin_tensor(torch.zeros(d_model, n_heads * head_dim, device=device, dtype=dtype_mapping[prefix + "q"]))
                    opt_layer[prefix + "k"] = pin_tensor(torch.zeros(d_model, n_kv_heads * head_dim, device=device, dtype=dtype_mapping[prefix + "k"]))
                    opt_layer[prefix + "v"] = pin_tensor(torch.zeros(d_model, n_kv_heads * head_dim, device=device, dtype=dtype_mapping[prefix + "v"]))
                    opt_layer[prefix + "o"] = pin_tensor(torch.zeros(n_heads * head_dim, d_model, device=device, dtype=dtype_mapping[prefix + "o"]))
                opt_layer[prefix + "ffn_norm"] = pin_tensor(torch.zeros(d_model, device=device, dtype=dtype_mapping[prefix + "ffn_norm"]))
                opt_layer[prefix + "router"] = pin_tensor(torch.zeros(d_model, num_experts, device=device, dtype=dtype_mapping[prefix + "router"]))
                if prefix == "o_m_" or not is_muon:
                    opt_layer[prefix + "up"] = pin_tensor(torch.zeros(num_experts, d_model, 2 * expert_dim, device=device, dtype=dtype_mapping[prefix + "up"]))
                    opt_layer[prefix + "down"] = pin_tensor(torch.zeros(num_experts, expert_dim, d_model, device=device, dtype=dtype_mapping[prefix + "down"]))
        else:
            cur_offset = 0
            for prefix in prefixes:
                opt_layer[prefix + "attn_norm"] = buffer[cur_offset:cur_offset + d_model * dtype_mapping[prefix + "attn_norm"].itemsize].view(dtype_mapping[prefix + "attn_norm"]).reshape(d_model)
                cur_offset += d_model * dtype_mapping[prefix + "attn_norm"].itemsize
                if prefix == "o_m_" or not is_muon:
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
                opt_layer[prefix + "router"] = buffer[cur_offset:cur_offset + d_model * num_experts * dtype_mapping[prefix + "router"].itemsize].view(dtype_mapping[prefix + "router"]).reshape(d_model, num_experts)
                cur_offset += d_model * num_experts * dtype_mapping[prefix + "router"].itemsize
                if prefix == "o_m_" or not is_muon: 
                    opt_layer[prefix + "up"] = buffer[cur_offset:cur_offset + num_experts * d_model * 2 * expert_dim * dtype_mapping[prefix + "up"].itemsize].view(dtype_mapping[prefix + "up"]).reshape(num_experts, d_model, 2 * expert_dim)
                    cur_offset += num_experts * d_model * 2 * expert_dim * dtype_mapping[prefix + "up"].itemsize
                    opt_layer[prefix + "down"] = buffer[cur_offset:cur_offset + num_experts * expert_dim * d_model * dtype_mapping[prefix + "down"].itemsize].view(dtype_mapping[prefix + "down"]).reshape(num_experts, expert_dim, d_model)
                    cur_offset += num_experts * expert_dim * d_model * dtype_mapping[prefix + "down"].itemsize
        
        return opt_layer, cur_offset

    # ## This is not well done because we only want to destory alloced mem, not if we created based on buffer...
    # def destroy(self):

    #     ## call destory_tensor() from mem_register.py to unregister...
    #     pass

    
    def init_weights(self, weights, std_factor=1.0):

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
        weights["w_q"].normal_(mean=0.0, std=model_std * std_factor)
        weights["w_k"].normal_(mean=0.0, std=model_std * std_factor)
        weights["w_v"].normal_(mean=0.0, std=model_std * std_factor)
        weights["w_up"].normal_(mean=0.0, std=model_std * std_factor)

        # 5. Output projections (WITH residual scaling)
        weights["w_o"].normal_(mean=0.0, std=attn_out_std * resid_scale * std_factor)
        weights["w_down"].normal_(mean=0.0, std=expert_std * resid_scale * std_factor)


        # 6. Router
        
        ## Unsure what to set for router weights...
        #router_std_factor = resid_scale
        router_std_factor = 1 / np.sqrt(self.model_dims["d_model"])
        weights["w_router"].normal_(mean=0.0, std=router_std_factor * std_factor)

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
        
        layer_id = self.layer_id

        ## ensure model path exists
        if not os.path.exists(model_path + f"/layers/{layer_id}"):
            os.makedirs(model_path + f"/layers/{layer_id}")

        for name, tensor in weights.items():
            torch.save(tensor.cpu(), model_path + f"/layers/{layer_id}/{name}.pt")
        
    
    def make_chunk_metadata(self, seq_lens, seq_positions, prior_seq_lens, prior_seq_offsets, device, local_layer_ids):

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

        expert_counts_host_dict = {}
        token_index_mapping_dict = {}

        num_experts = self.model_dims["num_routed_experts"]
        top_k = self.model_dims["top_k"]

        for local_layer_id in local_layer_ids:
            expert_counts_host_dict[local_layer_id] = torch.zeros(num_experts, dtype=torch.int32, device="cpu", pin_memory=True)
            token_index_mapping_dict[local_layer_id] = torch.zeros(total_q, top_k, dtype=torch.int32, device=device)

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
            "max_seqlen_k": max_seqlen_k,
            "expert_counts_host": expert_counts_host_dict,
            "token_index_mapping": token_index_mapping_dict,
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
        top_k = self.model_dims["top_k"]
        num_experts = self.model_dims["num_routed_experts"]

        if "attn_act_dtype" in self.model_hyperparams:
            attn_act_dtype = self.model_hyperparams["attn_act_dtype"]
        else:
            attn_act_dtype = torch.bfloat16

        if "ffn_act_dtype" in self.model_hyperparams:
            ffn_act_dtype = self.model_hyperparams["ffn_act_dtype"]
        else:
            ffn_act_dtype = torch.bfloat16

        act_slot = {}
        if buffer is None:

        
            act_slot["attn_norm_rstd"] = pin_tensor(torch.zeros(num_tokens, 1, device=device, dtype=torch.float32))
            act_slot["ffn_norm_rstd"] = pin_tensor(torch.zeros(num_tokens, 1, device=device, dtype=torch.float32))
            act_slot["x_inp"] = pin_tensor(torch.zeros(num_tokens, d_model, device=device, dtype=torch.bfloat16))
            act_slot["xk"] = pin_tensor(torch.zeros(num_tokens, n_kv_heads, head_dim, device=device, dtype=attn_act_dtype))
            act_slot["xv"] = pin_tensor(torch.zeros(num_tokens, n_kv_heads, head_dim, device=device, dtype=attn_act_dtype))
            ## maybe should make these fp32...?
            act_slot["x_router"] = pin_tensor(torch.zeros(num_tokens, num_experts, device=device, dtype=torch.bfloat16))
            act_slot["expert_counts"] = pin_tensor(torch.zeros(num_experts, device=device, dtype=torch.int32))
            act_slot["router_weights"] = pin_tensor(torch.zeros(num_tokens, top_k, device=device, dtype=torch.bfloat16))
            act_slot["chosen_experts"] = pin_tensor(torch.zeros(num_tokens, top_k, device=device, dtype=torch.int32))
            act_slot["scattered_router_weights"] = pin_tensor(torch.zeros(num_tokens * top_k, 1, device=device, dtype=torch.bfloat16))
            if saved_level >= 1:
                act_slot["attn_result"] = pin_tensor(torch.zeros(num_tokens, n_heads, head_dim, device=device, dtype=attn_act_dtype))
                act_slot["softmax_lse"] = pin_tensor(torch.zeros(n_heads, num_tokens, device=device, dtype=torch.float32))

            if saved_level >= 2:
                act_slot["xq"] = pin_tensor(torch.zeros(num_tokens, n_heads, head_dim, device=device, dtype=attn_act_dtype))
                act_slot["xo"] = pin_tensor(torch.zeros(num_tokens, d_model, device=device, dtype=attn_act_dtype))

            if saved_level >= 3:
                act_slot["x_up"] = pin_tensor(torch.zeros(num_tokens * top_k, 2 * expert_dim, device=device, dtype=ffn_act_dtype))


        else:

            cur_offset = 0

            attn_norm_rstd_size = num_tokens * torch.float32.itemsize
            ffn_norm_rstd_size = num_tokens * torch.float32.itemsize
            x_inp_size = num_tokens * d_model * attn_act_dtype.itemsize
            xk_size = num_tokens * n_kv_heads * head_dim * attn_act_dtype.itemsize
            xv_size = num_tokens * n_kv_heads * head_dim * attn_act_dtype.itemsize
            x_router_size = num_tokens * num_experts * torch.bfloat16.itemsize
            expert_counts_size = num_experts * torch.int32.itemsize
            router_weights_size = num_tokens * top_k * torch.bfloat16.itemsize
            chosen_experts_size = num_tokens * top_k * torch.int32.itemsize
            scattered_router_weights_size = num_tokens * top_k * torch.bfloat16.itemsize
            attn_result_size = num_tokens * n_heads * head_dim * attn_act_dtype.itemsize
            softmax_lse_size = n_heads * num_tokens * torch.float32.itemsize
            xq_size = num_tokens * n_heads * head_dim * attn_act_dtype.itemsize
            xo_size = num_tokens * d_model * attn_act_dtype.itemsize
            x_up_size = num_tokens * top_k * 2 * expert_dim * ffn_act_dtype.itemsize

            act_slot["attn_norm_rstd"] = buffer[:attn_norm_rstd_size].view(torch.float32).reshape(num_tokens, 1)
            cur_offset += attn_norm_rstd_size
            act_slot["ffn_norm_rstd"] = buffer[cur_offset:cur_offset + ffn_norm_rstd_size].view(torch.float32).reshape(num_tokens, 1)
            cur_offset += ffn_norm_rstd_size
            act_slot["x_inp"] = buffer[cur_offset:cur_offset + x_inp_size].view(attn_act_dtype).reshape(num_tokens, d_model)
            cur_offset += x_inp_size
            act_slot["xk"] = buffer[cur_offset:cur_offset + xk_size].view(attn_act_dtype).reshape(num_tokens, n_kv_heads, head_dim)
            cur_offset += xk_size
            act_slot["xv"] = buffer[cur_offset:cur_offset + xv_size].view(attn_act_dtype).reshape(num_tokens, n_kv_heads, head_dim)
            cur_offset += xv_size
            act_slot["x_router"] = buffer[cur_offset:cur_offset + x_router_size].view(torch.bfloat16).reshape(num_tokens, num_experts)
            cur_offset += x_router_size
            act_slot["expert_counts"] = buffer[cur_offset:cur_offset + expert_counts_size].view(torch.int32).reshape(num_experts)
            cur_offset += expert_counts_size
            act_slot["router_weights"] = buffer[cur_offset:cur_offset + router_weights_size].view(torch.bfloat16).reshape(num_tokens, top_k)
            cur_offset += router_weights_size
            act_slot["chosen_experts"] = buffer[cur_offset:cur_offset + chosen_experts_size].view(torch.int32).reshape(num_tokens, top_k)
            cur_offset += chosen_experts_size
            act_slot["scattered_router_weights"] = buffer[cur_offset:cur_offset + scattered_router_weights_size].view(torch.bfloat16).reshape(num_tokens * top_k, 1)
            cur_offset += scattered_router_weights_size
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
                act_slot["x_up"] = buffer[cur_offset:cur_offset + x_up_size].view(ffn_act_dtype).reshape(num_tokens * top_k, 2 * expert_dim)
                cur_offset += x_up_size

        total_size = 0
        for k, v in act_slot.items():
            total_size += v.numel() * v.dtype.itemsize

        return act_slot, total_size

    def get_act_slot_size(self, num_tokens, saved_level=None):

        total_size = 0

        if saved_level is None:
            saved_level = self.max_saved_activations_level

        resid_dtype = get_torch_dtype(self.model_dims["datatypes"]["residual"])
        router_dtype = get_torch_dtype(self.model_dims["datatypes"]["router"])

        ## assume non-router matmul activations are same datatype as resid
        attn_act_dtype = resid_dtype
        ffn_act_dtype = resid_dtype

        d_model = self.model_dims["d_model"]
        n_heads = self.model_dims["n_heads"]
        n_kv_heads = self.model_dims["n_kv_heads"]
        head_dim = self.model_dims["head_dim"]
        expert_dim = self.model_dims["expert_dim"]
        num_experts = self.model_dims["num_routed_experts"]
        top_k = self.model_dims["top_k"]

        ## Minimum saved level

        ## regardless of norm dtype, always saved rstd as fp32
        attn_norm_rstd_size = num_tokens * torch.float32.itemsize
        ffn_norm_rstd_size = num_tokens * torch.float32.itemsize
        x_inp_size = num_tokens * d_model * attn_act_dtype.itemsize
        xk_size = num_tokens * n_kv_heads * head_dim * attn_act_dtype.itemsize
        xv_size = num_tokens * n_kv_heads * head_dim * attn_act_dtype.itemsize
        x_router_size = num_tokens * num_experts * router_dtype.itemsize
        expert_counts_size = num_experts * torch.int32.itemsize
        router_weights_size = num_tokens * top_k * router_dtype.itemsize
        chosen_experts_size = num_tokens * top_k * torch.int32.itemsize
        scattered_router_weights_size = num_tokens * top_k * router_dtype.itemsize

        total_size += attn_norm_rstd_size + ffn_norm_rstd_size + x_inp_size + xk_size + xv_size + x_router_size + expert_counts_size + router_weights_size + chosen_experts_size + scattered_router_weights_size

        if saved_level == 0:
            return total_size
        
        
        attn_result_size = num_tokens * n_heads * head_dim * attn_act_dtype.itemsize
        softmax_lse_size = n_heads * num_tokens * torch.float32.itemsize

        total_size += attn_result_size + softmax_lse_size
        
        if saved_level == 1:
            return total_size

        
        xq_size = num_tokens * n_heads * head_dim * attn_act_dtype.itemsize
        xo_size = num_tokens * d_model * attn_act_dtype.itemsize

        total_size += xq_size + xo_size

        if saved_level == 2:
            return total_size

        x_up_size = num_tokens * top_k * 2 * expert_dim * ffn_act_dtype.itemsize

        total_size += x_up_size

        if saved_level == 3:
            return total_size

        return total_size

    def send_activations_home(self, home_act_slot, computed_act_slot, save_activations_level):

        save_level_mapping = {
            # START OF MIN SAVE (SAVED_LEVEL >= 0)
            "attn_norm_rstd": 0,
            "ffn_norm_rstd": 0,
            "x_inp": 0,
            "xk": 0,
            "xv": 0,
            "x_router": 0,
            "expert_counts": 0,
            "router_weights": 0,
            "chosen_experts": 0,
            "scattered_router_weights": 0,
            ## START OF PARTIAL SAVE (SAVED_LEVEL >= 1)
            "attn_result": 1,
            "softmax_lse": 1,
            ## START OF FULL SAVE (SAVED_LEVEL >= 2)
            "xq": 2,
            "xo": 2,
            "x_up": 3,
        }

        for k, v in computed_act_slot.items():

            if k not in save_level_mapping:
                continue

            save_level = save_level_mapping[k]

            if save_level < 0:
                continue

            if save_level <= save_activations_level:
                if k == "x_up":
                    home_up_act_slot = home_act_slot["x_up"]
                    home_up_offset = 0
                    num_experts = self.model_dims["num_routed_experts"]
                    # Iterate in consistent order (0, 1, 2, ...) to match fetch_activations
                    for expert_id in range(num_experts):
                        if expert_id in v:
                            expert_up = v[expert_id]
                            if expert_up.shape[0] > 0:
                                home_up_act_slot[home_up_offset:home_up_offset + expert_up.shape[0], :].copy_(expert_up, non_blocking=True)
                                home_up_offset += expert_up.shape[0]
                else:
                    home_act_slot[k].copy_(v, non_blocking=True)



    
    def fetch_activations(self, base_act_slot, home_act_slot, chunk_metadata, layer_id):

        act_slot = {}
        num_tokens = home_act_slot["x_inp"].shape[0]
        top_k = self.model_dims["top_k"]
        num_experts = self.model_dims["num_routed_experts"]

        ## use view of base act slot with the correct shape for this chunk
        for k, v in base_act_slot.items():
            if k not in home_act_slot:
                continue
            if k == "x_up":
                # Handle separately below
                continue
            if k == "expert_counts":
                act_slot[k] = v
            elif k == "scattered_router_weights":
                act_slot[k] = v[:num_tokens * top_k, :]
            elif k != "softmax_lse" and v.shape[0] != num_tokens:
                act_slot[k] = v[:num_tokens, :]
            elif k == "softmax_lse" and v.shape[1] != num_tokens:
                act_slot[k] = v[:, :num_tokens]
            else:
                act_slot[k] = v
        
        for k, v in home_act_slot.items():
            if k == "x_up":
                continue
            act_slot[k].copy_(v, non_blocking=True)

        # Reconstruct x_up as dict using expert_counts from metadata
        if "x_up" in home_act_slot:
            expert_counts_cpu = chunk_metadata["expert_counts_host"][layer_id]
            
            # Copy flat buffer to GPU
            gpu_x_up_flat = base_act_slot["x_up"][:num_tokens * top_k, :]
            gpu_x_up_flat.copy_(home_act_slot["x_up"], non_blocking=True)
            
            # Reconstruct dict with views into the flat buffer
            act_slot["x_up"] = {}
            cur_offset = 0
            for expert_id in range(num_experts):
                num_exp_tokens = expert_counts_cpu[expert_id].item()
                if num_exp_tokens > 0:
                    act_slot["x_up"][expert_id] = gpu_x_up_flat[cur_offset:cur_offset + num_exp_tokens, :]
                    cur_offset += num_exp_tokens

        return act_slot

        
    def fetch_weights(self, gpu_weights, cpu_weights):
        for k, v in gpu_weights.items():
            gpu_weights[k].copy_(cpu_weights[k], non_blocking=True)

    def get_fwd_flops(self, chunk_metadata):

        saved_fwd_flops = {}
        for i in range(self.max_saved_activations_level + 1):
            saved_fwd_flops[i] = 0

        num_tokens = chunk_metadata["total_q"]
        num_routed_experts = self.model_dims["num_routed_experts"]
        seq_lens = chunk_metadata["seq_lens_host"]
        prior_seq_lens = chunk_metadata["prior_seq_lens_host"]
        num_heads = self.model_dims["n_heads"]
        head_dim = self.model_dims["head_dim"]
        d_model = self.model_dims["d_model"]
        expert_dim = self.model_dims["expert_dim"]
        n_kv_heads = self.model_dims["n_kv_heads"]
        top_k = self.model_dims["top_k"]

        attn_dim = num_heads * head_dim
        ctx_dim = n_kv_heads * head_dim

        is_causal = self.model_dims["is_causal"]

        fwd_flops = 0

        ### have default "emergency" option be to save minimal amount of activations
        saved_fwd_flops[0] = 0

        for i in range(len(seq_lens)):

            ## here seq_len refers to only current portion of overall sequence if 
            ## sequence spans multiple chunks
            seq_len = seq_lens[i]
            prior_seq_len = prior_seq_lens[i]

            kv_flops = 2 * (2 * seq_len * d_model * ctx_dim)
            fwd_flops += kv_flops

            router_flops = 2 * seq_len * d_model * num_routed_experts
            fwd_flops += router_flops



            qo_flops = 2 * (2 * seq_len * d_model * attn_dim)
            saved_fwd_flops[2] += qo_flops
            saved_fwd_flops[3] += qo_flops
            
            fwd_flops += qo_flops

            ## 
            attn_flops_prior = 4 * seq_len * prior_seq_len * attn_dim

            ## caucal across seq len
            if is_causal:
                attn_flops_current = 2 * seq_len * seq_len * attn_dim
            else:
                attn_flops_current = 4 * seq_len * seq_len * attn_dim

            attn_flops = attn_flops_prior + attn_flops_current

            saved_fwd_flops[1] += attn_flops
            saved_fwd_flops[2] += attn_flops
            saved_fwd_flops[3] += attn_flops
            
            ## prior seq lens are full causal
            fwd_flops += attn_flops

            ## base matmuls for ffn

            up_flops = 2 * seq_len * d_model * top_k * 2 * expert_dim
            saved_fwd_flops[3] += up_flops

            fwd_flops += up_flops

            ### always save result of down proj ==> transition table
            down_flops = 2 * seq_len * d_model * top_k * expert_dim
            fwd_flops += down_flops
        
        return fwd_flops, saved_fwd_flops