import torch

import torch.cuda.nvtx as nvtx # Import NVTX
import numpy as np

from .ops import awsm_rmsnorm_fwd, awsm_rmsnorm_bwd, awsm_softmax, awsm_cross_entropy_loss, awsm_rmsnorm_bwd, awsm_adamw_step

class TransformerHead():

    def __init__(self, model_dims, model_hyperparams):
        self.model_dims = model_dims
        self.model_hyperparams = model_hyperparams

    def create(self, device = "cpu", pin_memory = True, is_grad=False, dtype_mapping = None):
        
        prefix = "w_"

        if is_grad:
            prefix = "g_"

        if dtype_mapping is None:
            dtype_mapping = {
                prefix + "final_norm": torch.bfloat16,
                prefix + "head_proj": torch.bfloat16,
            }

        if device != "cpu":
            pin_memory = False

        return {
            prefix + "final_norm": torch.zeros(self.model_dims["d_model"], device=device, dtype=dtype_mapping[prefix + "final_norm"], pin_memory=pin_memory),
            prefix + "head_proj": torch.zeros(self.model_dims["d_model"], self.model_dims["vocab_size"], device=device, dtype=dtype_mapping[prefix + "head_proj"], pin_memory=pin_memory),
        }

    def create_opt(self, device = "cpu", pin_memory = True, dtype_mapping = None):
        prefixes = ["o_m_", "o_v_"]

        if device != "cpu":
            pin_memory = False

        opt_layer = {}

        if dtype_mapping is None:
            dtype_mapping = {}
            for prefix in prefixes:
                dtype_mapping[prefix + "final_norm"] = torch.bfloat16
                dtype_mapping[prefix + "head_proj"] = torch.bfloat16

        for prefix in prefixes:
            opt_layer[prefix + "final_norm"] = torch.zeros(self.model_dims["d_model"], device=device, dtype=dtype_mapping[prefix + "final_norm"], pin_memory=pin_memory)
            opt_layer[prefix + "head_proj"] = torch.zeros(self.model_dims["d_model"], self.model_dims["vocab_size"], device=device, dtype=dtype_mapping[prefix + "head_proj"], pin_memory=pin_memory)

        return opt_layer

    def init_weights(self, weights, std_factor=1.0):
        # 1. Final Norm: Initialize to 1.0
        weights["w_final_norm"].fill_(1.0)

        # 2. Head Projection: Initialize with Truncated Normal
        ## Note: initializing with std = 1 / embed dim, not 1 / sqrt(d_model)
        #head_proj_std = 1 / self.model_dims["d_model"]
        head_proj_std = 1 / np.sqrt(self.model_dims["d_model"])
        weights["w_head_proj"].normal_(mean=0.0, std=head_proj_std)
        return weights

    def load(self, model_path, buffer=None, device = "cpu", pin_memory = True, is_opt=False, is_grad=False, dtype_mapping = None):
        new_layer = self.create(device = device, pin_memory = pin_memory, is_grad=is_grad, dtype_mapping = dtype_mapping)
        
        if is_opt:
            return self.load_opt(model_path, buffer)

        if model_path is None:
            if is_grad:
                return new_layer
            else:
                return self.init_weights(new_layer)

        prefix = "w_"

        if is_grad:
            prefix = "g_"

        weight_torch = torch.load(model_path + f"/head/{prefix}final_norm.pt", map_location="cpu")
        new_layer[prefix + "final_norm"].copy_(weight_torch, non_blocking=True)
        weight_torch = torch.load(model_path + f"/head/{prefix}head_proj.pt", map_location="cpu")
        new_layer[prefix + "head_proj"].copy_(weight_torch, non_blocking=True)
        return new_layer
    
    def load_opt(self, model_path, buffer):

        for name, tensor in buffer.items():
            weight_torch = torch.load(model_path + f"/head/{name}.pt")
            tensor.copy_(weight_torch)

        return

    def save(self, weights, model_path, is_grad=False, is_opt=False):
        ## weights are either params, grad or opt
        prefixes = ["w_"]
        if is_grad:
            prefixes = ["g_"]

        if is_opt:
            prefixes = ["o_m_", "o_v_"]
        
        for prefix in prefixes:
            torch.save(weights[prefix + "final_norm"].cpu(), model_path + f"/head/{prefix}final_norm.pt")
            torch.save(weights[prefix + "head_proj"].cpu(), model_path + f"/head/{prefix}head_proj.pt")

    def forward_backward(self, X, chunk_metadata, weights, labels, weight_grads, loss_scale_factor, head_chunk_size=1024, to_save_probs=False):
        dX, saved_probs = self.process(X, chunk_metadata, weights, head_chunk_size=head_chunk_size, labels=labels, weight_grads=weight_grads, loss_scale_factor=loss_scale_factor, to_save_probs=to_save_probs)
        return dX

    def step(self, weights, grad_weights, opt_state, opt_hyperparams):
        lr = opt_hyperparams["lr"]
        beta1 = opt_hyperparams["beta1"]
        beta2 = opt_hyperparams["beta2"]
        eps = opt_hyperparams["eps"]
        weight_decay = opt_hyperparams["weight_decay"]
        step_num = opt_hyperparams["step_num"]

        ret = awsm_adamw_step(weights["w_final_norm"], grad_weights["g_final_norm"], opt_state["o_m_final_norm"], opt_state["o_v_final_norm"], 
               lr=lr, beta1=beta1, beta2=beta2, eps=eps, 
               weight_decay=weight_decay, step=step_num)
        if ret != 0:
            print(f"AdamW Step failed for final norm at head")
            return -1
               
        ret = awsm_adamw_step(weights["w_head_proj"], grad_weights["g_head_proj"], opt_state["o_m_head_proj"], opt_state["o_v_head_proj"], 
               lr=lr, beta1=beta1, beta2=beta2, eps=eps, 
               weight_decay=weight_decay, step=step_num)
        if ret != 0:
            print(f"AdamW Step failed for head proj at head")
            return -1

        return 0

    def process(self, X, chunk_metadata, weights, head_chunk_size=1024, labels=None, weight_grads=None, loss_scale_factor=None, to_save_probs=False):
        chunk_tokens = X.shape[0]
        cur_head_token_offset = 0

        chunk_metadata["cur_head_token_offset"] = 0

        if to_save_probs:
            chunk_metadata["saved_probs"] = torch.empty(chunk_tokens, weights["w_head_proj"].shape[1], dtype=X.dtype, device=X.device)
        else:
            chunk_metadata["saved_probs"] = None

        chunk_metadata["per_token_loss"] = torch.empty(chunk_tokens, dtype=torch.float32, device=X.device)
        chunk_metadata["next_prediction"] = torch.empty(chunk_tokens, dtype=torch.int64, device=X.device)
        chunk_metadata["next_prediction_prob"] = torch.empty(chunk_tokens, dtype=torch.float32, device=X.device)

        head_chunk_count = 0
        
        while cur_head_token_offset < chunk_tokens:
            nvtx.range_push(f"Head Chunk: {head_chunk_count}")
            cur_head_num_tokens = min(head_chunk_size, chunk_tokens - cur_head_token_offset)
            
            if labels is None:
                chunk_labels = None
            else:
                chunk_labels = labels[cur_head_token_offset:cur_head_token_offset + cur_head_num_tokens]

            updated_X = self.process_head_chunk(X[cur_head_token_offset:cur_head_token_offset + cur_head_num_tokens], chunk_metadata, weights, 
                                                labels=chunk_labels, weight_grads=weight_grads, loss_scale_factor=loss_scale_factor, to_save_probs=to_save_probs)
            
            # returned newly alloced memory instead of same view
            if updated_X is not None:
                X[cur_head_token_offset:cur_head_token_offset + cur_head_num_tokens].copy_(updated_X)
                del updated_X

            cur_head_token_offset += cur_head_num_tokens
            chunk_metadata["cur_head_token_offset"] = cur_head_token_offset
            nvtx.range_pop()
            head_chunk_count += 1

        if labels is None and to_save_probs:
            return chunk_metadata["saved_probs"]

        final_loss = 0.0
        per_token_loss = chunk_metadata["per_token_loss"]

        #X_mean_loss = torch.mean(per_token_loss)
        dX = X
        
        return dX, chunk_metadata["saved_probs"]

    

    def process_head_chunk(self, X_slice, chunk_metadata, weights, labels=None, weight_grads=None, loss_scale_factor=None, to_save_probs=False):

        cur_head_num_tokens = X_slice.shape[0]
        cur_head_token_offset = chunk_metadata["cur_head_token_offset"]

        nvtx.range_push(f"RMSNorm Final")
        head_proj_in, final_norm_rstd = awsm_rmsnorm_fwd(X_slice, W=weights["w_final_norm"], rms_norm_eps=self.model_hyperparams["rms_norm_eps"])
        nvtx.range_pop()


        ## allocate new memory for logits
        nvtx.range_push(f"Matmul Head Proj")
        logits = torch.mm(head_proj_in, weights["w_head_proj"])
        nvtx.range_pop()

        ## temporary buffer
        head_probs = torch.empty(logits.shape, dtype=head_proj_in.dtype, device=head_proj_in.device)

        nvtx.range_push(f"Softmax")
        head_probs, head_max_idx, head_max_val = awsm_softmax(logits, out=head_probs, 
                                                    max_idx_out=chunk_metadata["next_prediction"][cur_head_token_offset:cur_head_token_offset + cur_head_num_tokens], 
                                                    max_val_out=chunk_metadata["next_prediction_prob"][cur_head_token_offset:cur_head_token_offset + cur_head_num_tokens])
        nvtx.range_pop()

        del logits

        if to_save_probs:
            chunk_metadata["saved_probs"][cur_head_token_offset:cur_head_token_offset + cur_head_num_tokens, :].copy_(head_probs)


        if labels is None:
            return None

        nvtx.range_push(f"Cross Entropy Loss")
        ## updates head_probs in-place
        dZ, loss = awsm_cross_entropy_loss(head_probs, labels, L=chunk_metadata["per_token_loss"][cur_head_token_offset:cur_head_token_offset + cur_head_num_tokens])
        nvtx.range_pop()

        del head_probs

        if weight_grads is None:
            return None
        
        nvtx.range_push(f"Compute Weight Gradients")

        dW = torch.addmm(weight_grads["g_head_proj"], head_proj_in.T, dZ, alpha=loss_scale_factor, beta=1.0, out=weight_grads["g_head_proj"])
        nvtx.range_pop()

        
        ## compute gradient dX_head_in

        nvtx.range_push(f"Compute dX_head_in")
        dX_head_in = torch.empty(head_proj_in.shape, dtype=head_proj_in.dtype, device=head_proj_in.device)

        del head_proj_in

        dX_head_in = torch.addmm(dX_head_in, dZ, weights["w_head_proj"].T, alpha=loss_scale_factor, beta=0.0, out=dX_head_in)
        nvtx.range_pop()
        
        ## allocate new memory for dX and can delete X after this in main loop
        nvtx.range_push(f"RMSNorm Backward")
        dX_slice, dW_norm, _ = awsm_rmsnorm_bwd(dX_head_in, X_slice, weights["w_final_norm"], final_norm_rstd, dW = weight_grads["g_final_norm"])
        nvtx.range_pop()

        
        del dX_head_in

        return dX_slice