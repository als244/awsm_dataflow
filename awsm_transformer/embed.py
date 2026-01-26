import torch
import numpy as np

from .ops import awsm_embedding_bwd, awsm_adamw_step

class TransformerEmbed():

    def __init__(self, model_dims, model_hyperparams):
        self.model_dims = model_dims
        self.model_hyperparams = model_hyperparams

    def forward(self, token_ids, weights):
        ## complex indexing creates a copy (which is what we want)
        vals = weights["w_tok_embeddings"][token_ids, :] 
        ## need to rescale by sqrt(d_model), so gradients are scaled correctly
        ## (if we use large init such as std=1, can skip this scaling)
        #vals *= np.sqrt(self.model_dims["d_model"])
        return vals 

    def backward(self, dX, token_ids, grad_weights):

        ### dX has shape (num_tokens, d_model), 
        ## and len(token_ids) == num_tokens
        ### we want to accumualte the gradients at rows corresponding token ids
        ## in grad_weights["g_tok_embeddings"]

        ## without scaling the backward gradient derivs are way too small and will
        ## end up roundning to 0, so we should scale by sqrt(d_model)

        #scale = np.sqrt(self.model_dims["d_model"])
        awsm_embedding_bwd(dX, token_ids, grad_weights["g_tok_embeddings"], scale=1.0)
        return None

    def step(self, weights, grad_weights, opt_state, opt_hyperparams):
        lr = opt_hyperparams["lr"]
        beta1 = opt_hyperparams["beta1"]
        beta2 = opt_hyperparams["beta2"]
        eps = opt_hyperparams["eps"]
        weight_decay = opt_hyperparams["weight_decay"]
        step_num = opt_hyperparams["step_num"]
        
        ret = awsm_adamw_step(weights["w_tok_embeddings"], grad_weights["g_tok_embeddings"], opt_state["o_m_tok_embeddings"], opt_state["o_v_tok_embeddings"], 
               lr=lr, beta1=beta1, beta2=beta2, eps=eps, 
               weight_decay=weight_decay, step=step_num)
        if ret != 0:
            print(f"AdamW Step failed for tok embeddings")
            return -1

        return 0

        
    def create(self, device = "cpu", pin_memory = True, is_grad=False, dtype_mapping = None):
        
        prefix = "w_"

        if is_grad:
            prefix = "g_"

        if dtype_mapping is None:
            dtype_mapping = {
                prefix + "tok_embeddings": torch.bfloat16,
            }

        if device != "cpu":
            pin_memory = False

        return {
            prefix + "tok_embeddings": torch.zeros(self.model_dims["vocab_size"], self.model_dims["d_model"], device=device, dtype=dtype_mapping[prefix + "tok_embeddings"], pin_memory=pin_memory),
        }
    
    def init_weights(self, weights, std_factor=1.0):
        ## PyTorch cenvention is .normal_ for embedding
        #weights["w_tok_embeddings"].normal_(mean=0.0, std=1.0)
        embed_std = 1.0
        #embed_std = 1 / np.sqrt(self.model_dims["d_model"])
        weights["w_tok_embeddings"].normal_(mean=0.0, std=std_factor * embed_std)

        return weights

    def create_opt(self, device = "cpu", pin_memory = True, dtype_mapping = None):
        prefixes = ["o_m_", "o_v_"]
        if device != "cpu":
            pin_memory = False

        opt_layer = {}

        if dtype_mapping is None:
            dtype_mapping = {}
            for prefix in prefixes:
                dtype_mapping[prefix + "tok_embeddings"] = torch.bfloat16

        for prefix in prefixes:
            opt_layer[prefix + "tok_embeddings"] = torch.zeros(self.model_dims["vocab_size"], self.model_dims["d_model"], device=device, dtype=dtype_mapping[prefix + "tok_embeddings"], pin_memory=pin_memory)

        return opt_layer

    def load(self, model_path, buffer=None, device = "cpu", pin_memory = True, is_opt=False, is_grad=False, dtype_mapping = None):

        if is_opt:
            return self.load_opt(model_path, buffer)

        new_layer = self.create(device = device, pin_memory = pin_memory, is_grad=is_grad, dtype_mapping = dtype_mapping)

        if model_path is None:
            if is_grad:
                return new_layer
            else:
                return self.init_weights(new_layer)

        prefix = "w_"

        if is_grad:
            prefix = "g_"

        weight_torch = torch.load(model_path + f"/embed/{prefix}tok_embeddings.pt", map_location="cpu")
        new_layer[prefix + "tok_embeddings"].copy_(weight_torch, non_blocking=True)
        return new_layer
    
    def load_opt(self, model_path, buffer):

        for name, tensor in buffer.items():
            weight_torch = torch.load(model_path + f"/embed/{name}.pt")
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
            torch.save(weights[prefix + "tok_embeddings"].cpu(), model_path + f"/embed/{prefix}tok_embeddings.pt")

