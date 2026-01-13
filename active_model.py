import torch
import torch.cuda.nvtx as nvtx
import ctypes
import numpy as np
import os
import sys
import gc

_cudart = ctypes.CDLL('libcudart.so')
_nvtxlib = ctypes.CDLL('libnvToolsExt.so')

class ActiveModel:
    def __init__(self, model_name, model_layers, working_set_config, local_config, hardware_env, chunk_metadata_func, embed_layer=None, head_layer=None, to_train=True, local_device="cuda:0", group_config=None):
        self.model_name = model_name
        self.working_set_config = working_set_config
        self.to_train = to_train
        self.step_count = 0

        ### dictionary from layer_id -> instance of layer_class
        self.model_layers = model_layers
        self.embed_layer = embed_layer
        self.head_layer = head_layer
        self.chunk_metadata_func = chunk_metadata_func
        self.hardware_env = hardware_env

        self.cpu_model_weights = {}
        self.cpu_grad_weights = {}
        self.zero_grad = True
        self.cpu_opt_weights = {}
        self.cpu_embed = {}
        self.cpu_head = {}
        self.cpu_act_slots = {}
        self.model_weights_gpu = {}
        self.grad_weights_gpu = {}
        self.opt_weights_gpu = {}
        self.act_slot_gpu = {}
        self.transitions_gpu = {}
        self.embed_gpu = {}
        self.head_gpu = {}
        self.device = local_device
        
        self.n_model_layers = len(model_layers)
        self.local_config = local_config
        self.local_rank = local_config["local_rank"]
        self.local_layer_ids = local_config["local_layer_ids"]
        self.local_layers = [model_layers[k] for k in self.local_layer_ids]
        

        self.n_gpu_model_layers = working_set_config["n_gpu_layers"]
        self.n_gpu_grads = working_set_config["n_gpu_grads"]
        self.n_gpu_opt_layers = working_set_config["n_gpu_opt_layers"]
        self.cpu_act_buffer_size = working_set_config["host_act_buffer_size"]
        self.gpu_act_buffer_size = working_set_config["gpu_act_buffer_size"]
        self.gpu_act_buffer = None

        self.max_seq_len = working_set_config["max_seq_len"]
        self.max_chunk_size = working_set_config["max_chunk_size"]
        self.target_round_tokens = working_set_config["target_round_tokens"]
        self.max_total_round_tokens = working_set_config["max_total_round_tokens"]

        self.max_host_mem_gb = working_set_config["max_host_mem_gb"]
        self.max_gpu_mem_gb = working_set_config["max_gpu_mem_gb"]

        self.used_home_mem_bytes = 0    
        self.used_gpu_mem_bytes = 0

        self.first_weight_layer_index_for_step = 0
        self.first_grad_layer_index_for_step = 0


    
        self.compute_stream = torch.cuda.Stream(device=local_device)
        self.inbound_stream = torch.cuda.Stream(device=local_device)
        self.outbound_stream = torch.cuda.Stream(device=local_device)
        self.inbound_fwd_context_stream = torch.cuda.Stream(device=local_device)
        try:
            _nvtxlib.nvtxNameCuStreamA(self.compute_stream.cuda_stream, b"Compute")
            _nvtxlib.nvtxNameCuStreamA(self.inbound_stream.cuda_stream, b"Inbound")
            _nvtxlib.nvtxNameCuStreamA(self.outbound_stream.cuda_stream, b"Outbound")
            _nvtxlib.nvtxNameCuStreamA(self.inbound_fwd_context_stream.cuda_stream, b"Inbound Fwd Context")
        except Exception as e:
            print(f"Error naming CUDA streams: {e}")

        self.weight_inbound_events = {}
        self.grad_weight_inbound_events = {}
        self.opt_inbound_events = {}
        
        ## during fwd pass

        ## mapping from act buffer index -> event
        self.act_slot_ready_events = {}

        ## mapping from (layer_id, chunk_id) -> event
        ## indicating that data is available in home act buffer for this chunk
        self.home_act_slot_available_events = {}

        ## during bwd pass for retrieval
        ## mapping from (layer_id, chunk_id) -> event
        self.inbound_act_slot_ready_events = {}

        ## mapping from (layer_id, chunk_id) -> index of act slot (still needs to be reshaped but data will be correct)
        self.dev_act_slot_mapping = {}

        self.profiler = nvtx




        ## TODO: Add group config
        self.group_config = group_config

        if group_config is not None:
            self.master_conn = group_config["master_conn"]
            self.next_conn = group_config["next_conn"]
            self.prev_conn = group_config["prev_conn"]
        else:
            self.master_conn = None
            self.next_conn = None
            self.prev_conn = None

    def initialize(self, save_path=None, to_overwrite=False):

        if save_path is not None:
            ## Check if save path exists
            if os.path.exists(save_path):
                if not to_overwrite:
                    print(f"Model path {save_path} already exists. Skipping initialization.")
                    return
                

        if self.embed_layer is not None:
            self.cpu_embed["weights"] = self.embed_layer.create()
            self.embed_layer.init_weights(self.cpu_embed["weights"])
        if self.head_layer is not None:
            self.cpu_head["weights"] = self.head_layer.create()
            self.head_layer.init_weights(self.cpu_head["weights"])

        for layer_id in self.local_layer_ids:
            layer = self.model_layers[layer_id]
            self.cpu_model_weights[layer_id] = layer.create()
            layer.init_weights(self.cpu_model_weights[layer_id])

        
        if save_path is not None:
            self.save(save_path, device_bookends=False)

        return

    def load(self, model_path, to_load_opt=False, to_sync=True):

        with self.inbound_stream:

            if self.embed_layer is not None:

                if "weights" not in self.cpu_embed:
                    self.cpu_embed["weights"] = self.embed_layer.load(model_path)
                

                self.embed_gpu["weights"] = self.embed_layer.create(device=self.device)
                
                for k, v in self.embed_gpu["weights"].items():
                    v.copy_(self.cpu_embed["weights"][k], non_blocking=True)


                if self.to_train:
                    self.cpu_embed["grad_weights"] = self.embed_layer.create(is_grad=True)
                    self.embed_gpu["grad_weights"] = self.embed_layer.create(device=self.device, is_grad=True)
                    self.cpu_embed["opt_state"] = self.embed_layer.create_opt()

                    if to_load_opt:
                        self.embed_layer.load(model_path, buffer=self.cpu_embed["opt_state"], is_opt=True)
                    
                    self.embed_gpu["opt_state"] = self.embed_layer.create_opt(device=self.device)
                    if to_load_opt:
                        for name, tensor in self.embed_gpu["opt_state"].items():
                            tensor.copy_(self.cpu_embed["opt_state"][name])


            if self.head_layer is not None:
                if "weights" not in self.cpu_head:
                    self.cpu_head["weights"] = self.head_layer.load(model_path)
            
                self.head_gpu["weights"] = self.head_layer.create(device=self.device)
                for k, v in self.head_gpu["weights"].items():
                    v.copy_(self.cpu_head["weights"][k], non_blocking=True)

                if self.to_train:
                    self.cpu_head["grad_weights"] = self.head_layer.create(is_grad=True)
                    self.cpu_head["opt_state"] = self.head_layer.create_opt()
                    
                    if to_load_opt:
                        self.head_layer.load(model_path, buffer=self.cpu_head["opt_state"], is_opt=True)

                    self.head_gpu["grad_weights"] = self.head_layer.create(device=self.device, is_grad=True)
                    self.head_gpu["opt_state"] = self.head_layer.create_opt(device=self.device)
                    
                    if to_load_opt:
                        for name, tensor in self.head_gpu["opt_state"].items():
                            tensor.copy_(self.cpu_head["opt_state"][name])
        

            ### 1. Load all layers into CPU memory

            cur_gpu_layer_cnt = 0
            cur_gpu_grad_cnt = 0

            for layer_id in self.local_layer_ids:
                layer = self.model_layers[layer_id]
                if layer_id not in self.cpu_model_weights:
                    self.cpu_model_weights[layer_id] = layer.load(model_path)

                if self.to_train:
                    self.cpu_grad_weights[layer_id] = layer.create(None, is_grad=True)
                    self.cpu_opt_weights[layer_id], _ = layer.create_opt(None)
                    
                    if to_load_opt:
                        layer.load(model_path, buffer=self.cpu_opt_weights[layer_id], is_opt=True)

                ### load the first n_gpu_model_layers into GPU memory
                if cur_gpu_layer_cnt < self.n_gpu_model_layers:
                    self.model_weights_gpu[cur_gpu_layer_cnt] = layer.create(device=self.device)
                    layer.fetch_weights(self.model_weights_gpu[cur_gpu_layer_cnt], self.cpu_model_weights[layer_id])
                    self.weight_inbound_events[layer_id] = self.inbound_stream.record_event()
                    cur_gpu_layer_cnt += 1
                
                if self.to_train and cur_gpu_grad_cnt < self.n_gpu_grads:
                    self.grad_weights_gpu[cur_gpu_grad_cnt] = layer.create(device=self.device, is_grad=True)
                    cur_gpu_grad_cnt += 1

            
            ## 2. creat gpu contexts/activations
            act_total_bytes = self.create_gpu_activations()
            self.gpu_act_total_bytes = act_total_bytes

            ## 3. create host act buffer

            ### BY default this uses cudaHostAlloc which only does reservations in powers of 2 (very bad)
            #self.cpu_act_buffer = torch.zeros(self.cpu_act_buffer_size, device="cpu", dtype=torch.uint8, pin_memory=True)

            ## Instead we create normal host memory and then register with GPU driver
            self.cpu_act_buffer = torch.zeros(self.cpu_act_buffer_size, device="cpu", dtype=torch.uint8)

            ret = _cudart.cudaHostRegister(ctypes.c_void_p(self.cpu_act_buffer.data_ptr()), ctypes.c_size_t(int(self.cpu_act_buffer_size)), ctypes.c_uint(0))
            if ret != 0:
                print(f"Failed to register host act buffer, with data ptr: {self.cpu_act_buffer.data_ptr()} and size of: {self.cpu_act_buffer_size}")
                return -1

            self.cpu_act_buffer_offset = 0

        if to_sync:
            torch.cuda.synchronize()

        return 0

    def save(self, model_path, save_opt_state=True, save_gradients=False, device_bookends=True):

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if self.embed_layer is not None:
            if not os.path.exists(model_path + "/embed"):
                os.makedirs(model_path + "/embed")
            if device_bookends:
                self.embed_layer.save(self.embed_gpu["weights"], model_path)
            else:
                self.embed_layer.save(self.cpu_embed["weights"], model_path)

        if self.head_layer is not None:
            if not os.path.exists(model_path + "/head"):
                os.makedirs(model_path + "/head")
            if device_bookends:
                self.head_layer.save(self.head_gpu["weights"], model_path)
            else:
                self.head_layer.save(self.cpu_head["weights"], model_path)

        if not os.path.exists(model_path + "/layers"):
            os.makedirs(model_path + "/layers")

        for layer_id in self.local_layer_ids:
            layer = self.model_layers[layer_id]
            if not os.path.exists(model_path + f"/layers/{layer_id}"):
                os.makedirs(model_path + f"/layers/{layer_id}")
            layer.save(self.cpu_model_weights[layer_id], model_path)
        
        if save_opt_state:
            self.save_opt_state(model_path, device_bookends=device_bookends)

        if save_gradients:
            self.save_gradients(model_path, device_bookends=device_bookends)

        return

    def destroy(self):

        ## clear host act/opt buffer
        ret = _cudart.cudaHostUnregister(ctypes.c_void_p(self.cpu_act_buffer.data_ptr()))
        if ret != 0:
            print(f"Failed to unregsiter host act buffer at addr: {self.cpu_act_buffer.data_ptr()} of size {self.cpu_act_buffer_size}")
        
        del self.cpu_act_buffer

        ## clear host model state

        ## TODO:

        # for layer_id, layer in self.cpu_model_weights:
        #     layer.destroy()

        # for layer_id, layer in self.cpu_grad_weights:
        #     layer.destroy()
        
        # for layer_id, layer in self.cpu_opt_weights:
        #     layer.destroy()

        ## clear up gpu memory

        self.clear_gpu_activations()
        self.clear_gpu_opt_state()

        for chunk_id, transition_tensor in self.transitions_gpu.items():
            del transition_tensor

        ## TODO: clear up rest of training state...

        return


    def save_gradients(self, model_path, device_bookends=True):

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if self.embed_layer is not None:
            if not os.path.exists(model_path + "/embed"):
                os.makedirs(model_path + "/embed")
            if device_bookends:
                self.embed_layer.save(self.embed_gpu["grad_weights"], model_path, is_grad=True)
            else:
                self.embed_layer.save(self.cpu_embed["grad_weights"], model_path, is_grad=True)

        if self.head_layer is not None:
            if not os.path.exists(model_path + "/head"):
                os.makedirs(model_path + "/head")
            if device_bookends:
                self.head_layer.save(self.head_gpu["grad_weights"], model_path, is_grad=True)
            else:
                self.head_layer.save(self.cpu_head["grad_weights"], model_path, is_grad=True)

        if not os.path.exists(model_path + "/layers"):
            os.makedirs(model_path + "/layers")

        for layer_id in self.local_layer_ids:
            layer = self.model_layers[layer_id]
            if not os.path.exists(model_path + f"/layers/{layer_id}"):
                os.makedirs(model_path + f"/layers/{layer_id}")
            layer.save(self.cpu_grad_weights[layer_id], model_path, is_grad=True)

        return

    def save_opt_state(self, model_path, device_bookends=True):

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if self.embed_layer is not None and "opt_state" in self.cpu_embed:
            if not os.path.exists(model_path + "/embed"):
                os.makedirs(model_path + "/embed")
            if device_bookends:
                self.embed_layer.save(self.embed_gpu["opt_state"], model_path, is_opt=True)
            else:
                self.embed_layer.save(self.cpu_embed["opt_state"], model_path, is_opt=True)

        if self.head_layer is not None and "opt_state" in self.cpu_head:
            if not os.path.exists(model_path + "/head"):
                os.makedirs(model_path + "/head")
            if device_bookends:
                self.head_layer.save(self.head_gpu["opt_state"], model_path, is_opt=True)
            else:
                self.head_layer.save(self.cpu_head["opt_state"], model_path, is_opt=True)

        if not os.path.exists(model_path + "/layers"):
            os.makedirs(model_path + "/layers")

        for layer_id in self.local_layer_ids:
            layer = self.model_layers[layer_id]
            if layer_id not in self.cpu_opt_weights:
                continue
            if not os.path.exists(model_path + f"/layers/{layer_id}"):
                os.makedirs(model_path + f"/layers/{layer_id}")
            layer.save(self.cpu_opt_weights[layer_id], model_path, is_opt=True)

        return


    def create_gpu_activations(self):

        total_bytes = 0

        ## 1. create contexts

        context_window_size = max(self.max_seq_len, self.max_chunk_size)
        n_kv_heads = self.model_layers[0].model_dims["n_kv_heads"]
        head_dim = self.model_layers[0].model_dims["head_dim"]

        self.fwd_context = {
            "k": torch.zeros(context_window_size, n_kv_heads, head_dim, device=self.device, dtype=torch.bfloat16),
            "v": torch.zeros(context_window_size, n_kv_heads, head_dim, device=self.device, dtype=torch.bfloat16),
        }

        self.bwd_context = {
            "dk": torch.zeros(context_window_size, n_kv_heads, head_dim, device=self.device, dtype=torch.bfloat16),
            "dv": torch.zeros(context_window_size, n_kv_heads, head_dim, device=self.device, dtype=torch.bfloat16),
        }

        total_bytes += self.fwd_context["k"].numel() * 2 + self.fwd_context["v"].numel() * 2 + self.bwd_context["dk"].numel() * 2 + self.bwd_context["dv"].numel() * 2


        ## 2. create blank act slots on GPU

        act_slot_size = self.model_layers[0].get_act_slot_size(self.max_chunk_size)

        self.n_gpu_act_slots = self.gpu_act_buffer_size // act_slot_size

        ## the first time this is called, create the gpu act buffer
        if self.gpu_act_buffer is None:
            self.gpu_act_buffer = torch.zeros(self.gpu_act_buffer_size, device=self.device, dtype=torch.uint8)
            print(f"# GPU Act Slots (Chunk Size: {self.max_chunk_size}): {self.n_gpu_act_slots}", flush=True)
        
        cur_gpu_act_buffer_offset = 0
        cur_gpu_act_buffer = self.gpu_act_buffer

        

        self.act_slot_ready_events.clear()
        self.act_slot_gpu.clear()

        for i in range(self.n_gpu_act_slots):
            new_act_slot, total_size = self.model_layers[0].make_act_slot(self.max_chunk_size, None, buffer=cur_gpu_act_buffer, device=self.device)
            self.act_slot_gpu[i] = new_act_slot
            self.act_slot_ready_events[i] = self.inbound_stream.record_event()
            total_bytes += total_size
            cur_gpu_act_buffer_offset += total_size
            cur_gpu_act_buffer = cur_gpu_act_buffer[total_size:]

        return total_bytes
    
    def clear_gpu_activations(self):

        fwd_keys = list(self.fwd_context.keys())
        bwd_keys = list(self.bwd_context.keys())

        for name in fwd_keys:
            tensor = self.fwd_context.pop(name)
            del tensor
        for name in bwd_keys:
            tensor = self.bwd_context.pop(name)
            del tensor

        return
            
            
    def create_gpu_opt_state(self):

        cur_gpu_opt_buffer_offset = 0
        cur_gpu_opt_buffer = self.gpu_act_buffer

        for i in range(self.n_gpu_opt_layers):
            layer_id = self.local_layer_ids[i]
            layer = self.model_layers[layer_id]
            self.opt_weights_gpu[i], total_size = layer.create_opt(buffer=cur_gpu_opt_buffer, device=self.device)
            
            cpu_opt_state = self.cpu_opt_weights[layer_id]
            with self.inbound_stream:
                for name, tensor in self.opt_weights_gpu[i].items():
                    tensor.copy_(cpu_opt_state[name], non_blocking=True)
            
            self.opt_inbound_events[layer_id] = self.inbound_stream.record_event()
            cur_gpu_opt_buffer_offset += total_size
            cur_gpu_opt_buffer = cur_gpu_opt_buffer[total_size:]

        return

    def clear_gpu_opt_state(self):

        for i in range(self.n_gpu_opt_layers):
            if i not in self.opt_weights_gpu:
                continue
            opt_dict = self.opt_weights_gpu.pop(i)
            opt_keys = list(opt_dict.keys())
            for name in opt_keys:
                tensor = opt_dict.pop(name)
                del tensor
            del opt_dict
        
        self.opt_inbound_events.clear()
    
        return

          
    def determine_saved_levels(self, seq_groups):

        total_chunks = sum([len(seq_group) for seq_group in seq_groups])

        self.n_home_act_slots = max(0, total_chunks * len(self.local_layer_ids) - self.n_gpu_act_slots)

        ## TODO: implement this for real (accounting for capacities, interconnect, processor speed)..
        
        slot_num = 0.
        saved_levels = {}
        for layer_id in self.local_layer_ids:
            cur_chunk_id = 0
            for seq_group in seq_groups:
                for chunk in seq_group:
                    if slot_num < self.n_home_act_slots:
                        ## FULL RECOMPUTE
                        saved_levels[(layer_id, cur_chunk_id)] = 0
                        ## PARTIAL RECOMPUTE (save attn)
                        #saved_levels[(layer_id, cur_chunk_id)] = 1
                        ## PARTIAL Recompute (save attn + xq,xo)
                        #saved_levels[(layer_id, cur_chunk_id)] = 2
                        ## FULLY SAVED
                        #saved_levels[(layer_id, cur_chunk_id)] = 3
                    else:
                        saved_levels[(layer_id, cur_chunk_id)] = -1
                    slot_num += 1
                    cur_chunk_id += 1
        return saved_levels

    def split_sequences(self, sequences):

        target_round_tokens = self.target_round_tokens
        max_total_round_tokens = self.max_total_round_tokens
        

        round_seqs = []
        cur_round_tokens = 0

        cur_round_seqs = []

        total_tokens = 0
        for seq in sequences:
            total_tokens += len(seq)
            if cur_round_tokens + len(seq) > target_round_tokens:
                ## package previous round and have this seq be part of new round
                if cur_round_tokens > 0:
                    round_seqs.append(cur_round_seqs) 
                
                if len(seq) > max_total_round_tokens:
                    raise ValueError(f"Sequence is too long: {len(seq)} tokens")
                cur_round_seqs = [seq]
                cur_round_tokens = len(seq)
                if len(seq) > target_round_tokens:
                    round_seqs.append(cur_round_seqs)
                    cur_round_tokens = 0
                    cur_round_seqs = []
            else:
                cur_round_tokens += len(seq)
                cur_round_seqs.append(seq)

        if len(cur_round_seqs) > 0:
            round_seqs.append(cur_round_seqs)

        return round_seqs, total_tokens

    def prepare_training_chunks(self, round_seqs):
        max_chunk_size = self.max_chunk_size
        
        # We will collect raw lists of tensors first to avoid repeated torch.cat() overhead
        final_chunks_data = [] 

        # Buffer for accumulating "Small" sequences
        # We only use this for sequences that fit within max_chunk_size
        cur_chunk_buf = {
            "tokens": [], "labels": [], 
            "lens": [], "pos": [], "prior_lens": [], "prior_offsets": [], "seqs_track": [],
            "current_size": 0
        }

        # Helper to flush the current buffer into the final list
        def flush_buffer():
            if cur_chunk_buf["current_size"] == 0:
                return
            
            # Save a copy of the current buffer state
            final_chunks_data.append({
                "tokens": list(cur_chunk_buf["tokens"]), # Shallow copy list
                "labels": list(cur_chunk_buf["labels"]),
                "lens": list(cur_chunk_buf["lens"]),
                "pos": list(cur_chunk_buf["pos"]),
                "prior_lens": list(cur_chunk_buf["prior_lens"]),
                "prior_offsets": list(cur_chunk_buf["prior_offsets"]),
                "seqs": list(cur_chunk_buf["seqs_track"])
            })
            
            # Reset buffer
            cur_chunk_buf["tokens"].clear()
            cur_chunk_buf["labels"].clear()
            cur_chunk_buf["lens"].clear()
            cur_chunk_buf["pos"].clear()
            cur_chunk_buf["prior_lens"].clear()
            cur_chunk_buf["prior_offsets"].clear()
            cur_chunk_buf["seqs_track"].clear()
            cur_chunk_buf["current_size"] = 0

        for s in round_seqs:
            s_len = len(s)
            
            # --- PATH A: Huge Sequence ---
            # Logic: If s is bigger than chunk size, it monopolizes every chunk it touches.
            if s_len > max_chunk_size:
                flush_buffer() # Clear any pending small sequences first
                
                cursor = 0
                while cursor < s_len:
                    # Take up to max_chunk_size
                    take = min(max_chunk_size, s_len - cursor)
                    
                    # Directly create a dedicated chunk for this slice
                    # We do NOT put this in the buffer, we go straight to final_chunks_data
                    # to ensure isolation.
                    final_chunks_data.append({
                        "tokens": [s.tokens[cursor : cursor + take]],
                        "labels": [s.targets[cursor : cursor + take]],
                        "lens": [take],
                        "pos": list(range(cursor, cursor + take)),
                        "prior_lens": [cursor],
                        "prior_offsets": [0],
                        "seqs": [[s, [cursor, cursor + take], [0, take]]]
                    })
                    cursor += take
                    
            # --- PATH B: Small Sequence ---
            # Logic: Try to fit in current buffer. If not, flush and start new buffer.
            else:
                # If adding this would overflow, flush the old buffer first
                if cur_chunk_buf["current_size"] + s_len > max_chunk_size:
                    flush_buffer()
                
                # Now append to buffer (it is guaranteed to fit because we flushed if needed)
                cur_chunk_buf["tokens"].append(s.tokens)
                cur_chunk_buf["labels"].append(s.targets)
                
                cur_chunk_buf["lens"].append(s_len)
                cur_chunk_buf["pos"].extend(range(s_len))
                cur_chunk_buf["prior_lens"].append(0)
                
                # Offset is the current size of the buffer before we add this sequence
                current_offset = cur_chunk_buf["current_size"]
                cur_chunk_buf["prior_offsets"].append(current_offset)
                cur_chunk_buf["seqs_track"].append([s, [0, s_len], [current_offset, current_offset + s_len]])
                
                cur_chunk_buf["current_size"] += s_len

        # Flush any remaining data in buffer
        flush_buffer()

        # --- Final Processing ---
        # Now we do the heavy lifting (GPU transfers, metadata calc) in one pass
        my_chunks = []
        
        for i, data in enumerate(final_chunks_data):
            # Concatenate tensors ONCE per chunk
            chunk_tokens = torch.cat(data["tokens"]).long().to(self.device, non_blocking=True)
            chunk_labels = torch.cat(data["labels"]).long().to(self.device, non_blocking=True)
            
            chunk_metadata = self.chunk_metadata_func(data["lens"], data["pos"], data["prior_lens"], data["prior_offsets"], self.device, self.local_layer_ids)
            
            my_chunks.append({
                "id": i, 
                "chunk_metadata": chunk_metadata, 
                "chunk_token_ids": chunk_tokens, 
                "chunk_label_ids": chunk_labels, 
                "chunk_seqs": data["seqs"]
            })

        # --- Reconstruct Sequence Groups ---
        # A new group starts whenever a chunk contains the START of a sequence (prior_len == 0).
        seq_groups = []
        current_group = []
        
        for chunk_idx, chunk in enumerate(my_chunks):
            # Access the raw metadata we stored to check if this is a group start
            # The first sequence in the chunk determines if it's a new group
            start_of_sequence_offset = final_chunks_data[chunk_idx]["prior_lens"][0]
            
            # If offset is 0, it means a sequence starts at the beginning of this chunk.
            # This signals a new group.
            if start_of_sequence_offset == 0:
                if current_group:
                    seq_groups.append(current_group)
                current_group = []
                
            current_group.append(chunk)
            
        if current_group:
            seq_groups.append(current_group)

        chunk_mapping = {chunk["id"]: chunk for chunk in my_chunks}
        
        return seq_groups, chunk_mapping

    def start_profile(self):
        return _cudart.cudaProfilerStart()
    
    def stop_profile(self):
        return _cudart.cudaProfilerStop()

    def update_fwd_context(self, completed_seq_group_ind, completed_chunk_group_ind, completed_layer_ind, seq_groups, fwd_context):

        if completed_layer_ind == 0 and completed_seq_group_ind == 0:
            return

        
        next_chunk_to_update = None
        ## find chunk in prior seq group at same group index, otherwise prior layer...

        if completed_seq_group_ind > 0:
            next_seq_group_ind = completed_seq_group_ind - 1
            while next_seq_group_ind >= 0:
                seq_group = seq_groups[next_seq_group_ind]
                if len(seq_group) > completed_chunk_group_ind:
                    ## update this chunk
                    next_chunk_to_update = seq_group[completed_chunk_group_ind]
                    next_to_update_layer_id = self.local_layer_ids[completed_layer_ind]
                    break
                next_seq_group_ind -= 1

        if next_chunk_to_update is None:
            ## update prior layer
            cur_seq_group_ind = len(seq_groups) - 1
            cur_layer_ind = completed_layer_ind - 1

            if cur_layer_ind < 0:
                return

            while cur_seq_group_ind >= 0:
                seq_group = seq_groups[cur_seq_group_ind]
                if len(seq_group) > completed_chunk_group_ind:
                    ## update this chunk
                    next_chunk_to_update = seq_group[completed_chunk_group_ind]
                    next_to_update_layer_id = self.local_layer_ids[cur_layer_ind]
                    break
                cur_seq_group_ind -= 1
        

        if next_chunk_to_update is not None:


            chunk_metadata = next_chunk_to_update["chunk_metadata"]

            ## first check if we already have the activations on device

          
            ## at time of dispatching we know that at this the point in which this function is called
            ## we have already enqueued data transfer of activations for this required chunk
            ## so we can wait on this event

            ## otherwise (else condition) we are likely in very low mem or long-context scenario where # gpu act slots < chunks per sequence
            ## and so the required fwd context has not been enqueued yet, meaning we should transfer data directly from home
            if (next_to_update_layer_id, next_chunk_to_update["id"]) in self.inbound_act_slot_ready_events:
                self.inbound_fwd_context_stream.wait_event(self.inbound_act_slot_ready_events[(next_to_update_layer_id, next_chunk_to_update["id"])])

                fwd_act = self.dev_act_slot_mapping[(next_to_update_layer_id, next_chunk_to_update["id"])]
                

                ## should either be 1 chunk with multiple sequences packed
                ## or multiple chunks of contiguous sequence. Thus replacing
                ## contiguous slab of fwd context at corresponding positions is correct...
                ## chunk group ind represents position within fwd context window...
                
                """
                cur_seq_offset = 0
                for seq_idx in range(len(chunk_metadata["seq_lens_host"])):
                    prior_start_idx = chunk_metadata["prior_seq_offsets_host"][seq_idx]
                    prior_end_idx = prior_start_idx + chunk_metadata["prior_seq_lens_host"][seq_idx]

                    new_seqlen = chunk_metadata["seq_lens_host"][seq_idx]
                    new_end_idx = prior_end_idx + new_seqlen

                    ## copy k and v to context windows...

                    with self.inbound_fwd_context_stream:
                        self.profiler.range_push(f"Update Fwd Context (from Local): Layer {next_to_update_layer_id}, Chunk {next_chunk_to_update['id']}")
                        fwd_context["k"][prior_end_idx:new_end_idx, :].copy_(fwd_act["xk"][cur_seq_offset:cur_seq_offset + new_seqlen, :])
                        fwd_context["v"][prior_end_idx:new_end_idx, :].copy_(fwd_act["xv"][cur_seq_offset:cur_seq_offset + new_seqlen, :])
                        self.profiler.range_pop()

                    cur_seq_offset += new_seqlen
                """
                
                start_chunk_idx = chunk_metadata["prior_seq_offsets_host"][0] + chunk_metadata["prior_seq_lens_host"][0]
                total_q = chunk_metadata["total_q"]
                with self.inbound_fwd_context_stream:
                    fwd_context["k"][start_chunk_idx: start_chunk_idx + total_q, :].copy_(fwd_act["xk"])
                    fwd_context["v"][start_chunk_idx: start_chunk_idx + total_q, :].copy_(fwd_act["xv"])

            else:
                ## otherwise load directly from cpu_act_buffer
                ## need to wait to ensure data is available as maybe we are mid transfer and it is hard to locate data
                self.inbound_fwd_context_stream.wait_event(self.home_act_slot_available_events[(next_to_update_layer_id, next_chunk_to_update["id"])])

                home_act_slot = self.cpu_act_slots[(next_to_update_layer_id, next_chunk_to_update["id"])]

                """
                cur_seq_offset = 0
                for seq_idx in range(len(chunk_metadata["seq_lens_host"])):
                    prior_start_idx = chunk_metadata["prior_seq_offsets_host"][seq_idx]
                    prior_end_idx = prior_start_idx + chunk_metadata["prior_seq_lens_host"][seq_idx]

                    new_seqlen = chunk_metadata["seq_lens_host"][seq_idx]
                    new_end_idx = prior_end_idx + new_seqlen

                    ## copy k and v to context windows...
                    with self.inbound_fwd_context_stream:
                        self.profiler.range_push(f"Update Fwd Context (from Home): Layer {next_to_update_layer_id}, Chunk {next_chunk_to_update['id']}")
                        fwd_context["k"][prior_end_idx:new_end_idx, :].copy_(home_act_slot["xk"][cur_seq_offset:cur_seq_offset + new_seqlen, :])
                        fwd_context["v"][prior_end_idx:new_end_idx, :].copy_(home_act_slot["xv"][cur_seq_offset:cur_seq_offset + new_seqlen, :])
                        self.profiler.range_pop()

                    cur_seq_offset += new_seqlen
                """    

                start_chunk_idx = chunk_metadata["prior_seq_offsets_host"][0] + chunk_metadata["prior_seq_lens_host"][0]
                total_q = chunk_metadata["total_q"]
                with self.inbound_fwd_context_stream:
                    fwd_context["k"][start_chunk_idx: start_chunk_idx + total_q, :].copy_(home_act_slot["xk"])
                    fwd_context["v"][start_chunk_idx: start_chunk_idx + total_q, :].copy_(home_act_slot["xv"])
        return

    def fwd_bwd(self, sequences, loss_scale_factor=None, verbose=True):
    
        all_round_seqs, total_tokens = self.split_sequences(sequences)


        if verbose:
            print(f"\n[Step {self.step_count + 1}] Processing {len(sequences)} Sequences, {total_tokens} Total Tokens\n\t{len(all_round_seqs)} Gradient Accumulation Rounds", flush=True)

        round_num = 0

        if loss_scale_factor is None:
            loss_scale_factor = 1.0

        for round_seqs in all_round_seqs:

            self.home_act_slot_available_events.clear()
            self.inbound_act_slot_ready_events.clear()
            self.dev_act_slot_mapping.clear()
            self.grad_weight_inbound_events.clear()

            with self.compute_stream:
                self.profiler.range_push("Fwd+Bwd Round")

            if verbose:
                print(f"\tRound {round_num+1}/{len(all_round_seqs)}\n\t{len(round_seqs)} Sequences, {sum([len(s) for s in round_seqs])} Tokens", flush=True)

            with self.compute_stream:
                self.profiler.range_push("Prepare Training Chunks")
                seq_groups, chunk_mapping = self.prepare_training_chunks(round_seqs)
                saved_levels = self.determine_saved_levels(seq_groups)
                
                if verbose:
                    print(f"\t\t{len(seq_groups)} Seq Groups\n\t\t{len(chunk_mapping)} Chunks\n", flush=True)           
                
                self.cpu_cur_act_buffer = self.cpu_act_buffer
                self.cpu_cur_act_buffer_offset = 0
                for k, v in saved_levels.items():
                    layer_id, chunk_id = k
                    if v == -1:
                        self.cpu_act_slots[(layer_id, chunk_id)] = None
                        continue
                    layer = self.model_layers[layer_id]
                    chunk = chunk_mapping[chunk_id]
                    act_slot, total_bytes = layer.make_act_slot(len(chunk["chunk_token_ids"]), v, buffer=self.cpu_cur_act_buffer)
                    self.cpu_act_slots[(layer_id, chunk_id)] = act_slot
                    self.cpu_cur_act_buffer_offset += total_bytes
                    self.cpu_cur_act_buffer = self.cpu_act_buffer[self.cpu_cur_act_buffer_offset:]

                ## initialize transitions
                if self.embed_layer is not None:
                    
                    for chunk_id, chunk in chunk_mapping.items():
                        chunk_metadata = chunk["chunk_metadata"]
                        chunk_token_ids = chunk["chunk_token_ids"]
                        self.transitions_gpu[chunk_id] = self.embed_layer.forward(chunk_token_ids, self.embed_gpu["weights"])
                        #torch.cuda.synchronize()
                        #torch.save(self.transitions_gpu[chunk_id], "fineweb_ckpts/my_compare_moe/acts/fwd_embed.pt")
               
                self.profiler.range_pop()


            with self.compute_stream:
                self.profiler.range_push("Forward")


            cur_act_slot_idx = 0
            cur_weight_idx = 0
            cur_grad_weight_idx = self.n_gpu_grads - 1
            
            if len(self.local_layer_ids) - self.n_gpu_grads - 1 >= 0:
                next_grad_layer_id = self.local_layer_ids[len(self.local_layer_ids) - self.n_gpu_grads - 1]
            else:
                next_grad_layer_id = -1

            
            prefetched_grads = False

            ## prefetching activations during backward pass
            next_act_slot_prefetch = (-1, -1)

            total_chunks = sum([len(seq_group) for seq_group in seq_groups])

            for k_ind in range(len(self.local_layer_ids)):

                k = self.local_layer_ids[k_ind]

                layer = self.model_layers[k]

                self.compute_stream.wait_event(self.weight_inbound_events[k])

                with self.compute_stream:
                    self.profiler.range_push(f"Layer {k}")


                cur_chunk_id = 0

                for seq_group in seq_groups:

                    for chunk in seq_group:

                        self.compute_stream.wait_event(self.act_slot_ready_events[cur_act_slot_idx])

                        ## TODO: wait upon transition ready (for multi-GPU)

                        with self.compute_stream:
                            self.profiler.range_push(f"Chunk {cur_chunk_id}")
                            self.transitions_gpu[cur_chunk_id], computed_act_slot = layer.forward(self.transitions_gpu[cur_chunk_id], chunk["chunk_metadata"], self.model_weights_gpu[cur_weight_idx], self.act_slot_gpu[cur_act_slot_idx], self.fwd_context)
                            self.profiler.range_pop()

                            #torch.cuda.synchronize()
                            #torch.save(self.transitions_gpu[chunk_id], f"fineweb_ckpts/my_compare_moe/acts/fwd_layer_{k}.pt")
                    
                        ## send activations home
                        saved_level = saved_levels[(k, cur_chunk_id)]
                        if saved_level != -1:

                            self.outbound_stream.wait_stream(self.compute_stream)

                            with self.outbound_stream:
                                self.profiler.range_push(f"Save Activations: Layer {k}, Chunk {cur_chunk_id}")
                                layer.send_activations_home(self.cpu_act_slots[(k, cur_chunk_id)], computed_act_slot, saved_level)
                                self.profiler.range_pop()
                            self.act_slot_ready_events[cur_act_slot_idx] = self.outbound_stream.record_event()
                            self.home_act_slot_available_events[(k, cur_chunk_id)] = self.outbound_stream.record_event()
                            next_act_slot_prefetch = (k_ind, cur_chunk_id)
                        else:
                            self.act_slot_ready_events[cur_act_slot_idx] = self.compute_stream.record_event()
                            self.inbound_act_slot_ready_events[(k, cur_chunk_id)] = self.compute_stream.record_event()
                            self.dev_act_slot_mapping[(k, cur_chunk_id)] = computed_act_slot
                            self.home_act_slot_available_events[(k, cur_chunk_id)] = None

                        ### advance next act slot unless final computation (final layer + final chunk)
                        if k_ind < len(self.local_layer_ids) - 1 or cur_chunk_id < total_chunks - 1:
                            cur_act_slot_idx = (cur_act_slot_idx + 1) % self.n_gpu_act_slots

                        cur_chunk_id += 1
                        
                        
                ## prefetch next layer
                if k_ind + self.n_gpu_model_layers < len(self.local_layer_ids):
                    self.inbound_stream.wait_stream(self.compute_stream)

                    next_layer_id = self.local_layer_ids[k_ind + self.n_gpu_model_layers]

                    with self.inbound_stream:
                        self.profiler.range_push(f"Prefetch Weights: Layer {next_layer_id}")
                        layer.fetch_weights(self.model_weights_gpu[cur_weight_idx], self.cpu_model_weights[next_layer_id])
                        self.profiler.range_pop()

                    self.weight_inbound_events[next_layer_id] = self.inbound_stream.record_event()
                    self.weight_inbound_events[k] = None
                else:
                    if not prefetched_grads:
                        self.inbound_stream.wait_stream(self.compute_stream)
                        with self.inbound_stream:
                            for grad_prefetch_num in range(self.n_gpu_grads):
                                grad_layer_id = self.local_layer_ids[len(self.local_layer_ids) - grad_prefetch_num - 1]
                                grad_layer = self.model_layers[grad_layer_id]

                                if self.zero_grad:
                                    self.profiler.range_push(f"Initializing Zero-Gradients: Layer {grad_layer_id}")
                                    for name, tensor in self.grad_weights_gpu[self.n_gpu_grads - grad_prefetch_num - 1].items():
                                            tensor.zero_()
                                    self.profiler.range_pop()
                                else:
                                    self.profiler.range_push(f"Prefetch Gradients: Layer {grad_layer_id}")
                                    grad_layer.fetch_weights(self.grad_weights_gpu[self.n_gpu_grads - grad_prefetch_num - 1], self.cpu_grad_weights[grad_layer_id])
                                    self.profiler.range_pop()
                                self.grad_weight_inbound_events[grad_layer_id] = self.inbound_stream.record_event()
                                
                        prefetched_grads = True
                
                ## if the last layer then dont need to update and this will be the first layer index in backward
                if k_ind < len(self.local_layer_ids) - 1:
                    cur_weight_idx = (cur_weight_idx + 1) % self.n_gpu_model_layers

                ### pop layer
                with self.compute_stream:
                    self.profiler.range_pop()

            #print(f"\n\n\nFORWARD PASS COMPLETE\n\n\n", flush=True)

            with self.compute_stream:
                self.profiler.range_pop()
            

            if self.head_layer is not None:
                with self.compute_stream:
                    
                    self.profiler.range_push("Head")

                    if self.zero_grad:
                        for name, tensor in self.head_gpu["grad_weights"].items():
                            tensor.zero_()

                    for j in range(len(chunk_mapping)):
                        chunk = chunk_mapping[j]
                        chunk_metadata = chunk["chunk_metadata"]
                        chunk_token_ids = chunk["chunk_token_ids"]
                        chunk_label_ids = chunk["chunk_label_ids"]

                        
                            
                        #loss_scale_factor = 1.0
                        self.transitions_gpu[j] = self.head_layer.forward_backward(self.transitions_gpu[j], chunk_metadata, self.head_gpu["weights"], chunk_label_ids, self.head_gpu["grad_weights"], loss_scale_factor)
                        #torch.cuda.synchronize()
                        #torch.save(self.transitions_gpu[j], f"fineweb_ckpts/my_compare_moe/acts/bwd_head.pt")
                        
                        ## copy chunk_metadata["per_token_loss"] in seq objects
                        for seq_info in chunk["chunk_seqs"]:
                            s, seq_range, chunk_range = seq_info
                            s.per_token_loss[seq_range[0]:seq_range[1]].copy_(chunk_metadata["per_token_loss"][chunk_range[0]:chunk_range[1]])
                    self.profiler.range_pop()

                
                ## send gradient home (optional duplication for safekeeping)
                """
                self.outbound_stream.wait_stream(self.compute_stream)
                with self.outbound_stream:
                    self.profiler.range_push("Send Gradients Home: Head")
                    for name, tensor in self.head_gpu["grad_weights"].items():
                        self.cpu_head["grad_weights"][name].copy_(tensor, non_blocking=True)
                    self.profiler.range_pop()
                """
          
            with self.compute_stream:
                self.profiler.range_push("Backward")

            cur_grad_weight_idx = self.n_gpu_grads - 1

            for k_ind in range(len(self.local_layer_ids) - 1, -1, -1):

                k = self.local_layer_ids[k_ind]
                layer = self.model_layers[k]

                with self.compute_stream:
                    self.profiler.range_push(f"Layer {k}")


                ### Wait on model parameters
                self.compute_stream.wait_event(self.weight_inbound_events[k])
                
                ## Wait on model gradients
                self.compute_stream.wait_event(self.grad_weight_inbound_events[k])

                cur_chunk_id = total_chunks - 1

                for seq_group_ind in range(len(seq_groups) - 1, -1, -1):
                        
                    seq_group = seq_groups[seq_group_ind]

                    ### Wait on forward context
                    ## guaranteed that the next seq group will be available
                    self.compute_stream.wait_stream(self.inbound_fwd_context_stream)
                        
                    for seq_group_chunk_ind in range(len(seq_group) - 1, -1, -1):

                        chunk = seq_group[seq_group_chunk_ind]

                        ### Wait on act slot
                        self.compute_stream.wait_event(self.inbound_act_slot_ready_events[(k, cur_chunk_id)])

                        ## dictionary populated with items saved before, 
                        # (dict with items was created up fetching activations; and now the buffers are ready
                        dev_act_slot = self.dev_act_slot_mapping[(k, cur_chunk_id)]

                        with self.compute_stream:
                            self.profiler.range_push(f"Chunk {cur_chunk_id} (Recompute)")

                            layer.forward_recompute(dev_act_slot, self.act_slot_gpu[cur_act_slot_idx], chunk["chunk_metadata"], self.model_weights_gpu[cur_weight_idx], self.fwd_context)

                            self.profiler.range_pop()

                            self.profiler.range_push(f"Chunk {cur_chunk_id} (Backward)")
                            self.transitions_gpu[cur_chunk_id] = layer.backward(self.transitions_gpu[cur_chunk_id], chunk["chunk_metadata"], self.model_weights_gpu[cur_weight_idx], self.grad_weights_gpu[cur_grad_weight_idx], dev_act_slot, self.fwd_context, self.bwd_context)
                            self.profiler.range_pop()

                            #torch.cuda.synchronize()
                            #torch.save(self.transitions_gpu[chunk_id], f"fineweb_ckpts/my_compare_moe/acts/bwd_layer_{k}.pt")

                        
                            
                        ### update forward context for prior seq group...
                        self.inbound_fwd_context_stream.wait_stream(self.compute_stream)
                        self.update_fwd_context(seq_group_ind, seq_group_chunk_ind, k_ind, seq_groups, self.fwd_context)
                        
                        ## prefetch next activation slot...
                        if next_act_slot_prefetch[0] != -1:
                            next_act_pre_layer_ind, next_act_pre_chunk_id = next_act_slot_prefetch
                            next_act_pre_layer_id = self.local_layer_ids[next_act_pre_layer_ind]
                            self.inbound_stream.wait_stream(self.compute_stream)
                            self.inbound_stream.wait_event(self.home_act_slot_available_events[(next_act_pre_layer_id, next_act_pre_chunk_id)])
                            with self.inbound_stream:
                                self.profiler.range_push(f"Prefetch Activations: Layer {next_act_pre_layer_id}, Chunk {next_act_pre_chunk_id}")
                                self.dev_act_slot_mapping[(next_act_pre_layer_id, next_act_pre_chunk_id)] = layer.fetch_activations(self.act_slot_gpu[cur_act_slot_idx], self.cpu_act_slots[(next_act_pre_layer_id, next_act_pre_chunk_id)], chunk_mapping[next_act_pre_chunk_id]["chunk_metadata"],
    next_act_pre_layer_id)
                                self.inbound_act_slot_ready_events[(next_act_pre_layer_id, next_act_pre_chunk_id)] = self.inbound_stream.record_event()
                                self.profiler.range_pop()
                            
                            if next_act_pre_chunk_id > 0:
                                next_act_slot_prefetch = (next_act_pre_layer_ind, next_act_pre_chunk_id - 1)
                            else:
                                if next_act_pre_layer_ind > 0:
                                    next_act_slot_prefetch = (next_act_pre_layer_ind - 1, total_chunks - 1)
                                else:
                                    next_act_slot_prefetch = (-1, -1)
                        
                        ## work the reverse direction during bwd
                        if cur_act_slot_idx == 0:
                            cur_act_slot_idx = self.n_gpu_act_slots - 1
                        else:
                            cur_act_slot_idx -= 1

                        cur_chunk_id -= 1
                                    
                
                ## send gradients
                self.outbound_stream.wait_stream(self.compute_stream)
                with self.outbound_stream:
                    self.profiler.range_push(f"Send Gradients Home: Layer {k}")
                    for name, tensor in self.grad_weights_gpu[cur_grad_weight_idx].items():
                        self.cpu_grad_weights[k][name].copy_(tensor, non_blocking=True)
                    self.profiler.range_pop()
                
                ## prefetch next weights
                if k_ind - self.n_gpu_model_layers >= 0:
                    self.inbound_stream.wait_stream(self.compute_stream)

                    next_layer_id = self.local_layer_ids[k_ind - self.n_gpu_model_layers]

                    with self.inbound_stream:
                        self.profiler.range_push(f"Prefetch Weights: Layer {next_layer_id}")
                        layer.fetch_weights(self.model_weights_gpu[cur_weight_idx], self.cpu_model_weights[next_layer_id])
                        self.weight_inbound_events[next_layer_id] = self.inbound_stream.record_event()
                        self.weight_inbound_events[k] = None
                        self.profiler.range_pop()
                    
                    if k_ind - self.n_gpu_model_layers == 0:
                        self.first_weight_layer_index_for_step = cur_weight_idx
                
                if cur_weight_idx == 0:
                    cur_weight_idx = self.n_gpu_model_layers - 1
                else:
                    cur_weight_idx -= 1

                ## prefetch next grad weights
                if k_ind - self.n_gpu_grads >= 0:
                    self.inbound_stream.wait_stream(self.compute_stream)
                    ## make sure gradients finished being sent home before retrieving next gradients
                    self.inbound_stream.wait_stream(self.outbound_stream)
                    next_grad_layer_id = self.local_layer_ids[k_ind - self.n_gpu_grads]
                    with self.inbound_stream:

                        if self.zero_grad:
                            self.profiler.range_push(f"Initializing Zero-Gradients: Layer {next_grad_layer_id}")
                            for name, tensor in self.grad_weights_gpu[cur_grad_weight_idx].items():
                                    tensor.zero_()
                            self.profiler.range_pop()
                        else:
                            self.profiler.range_push(f"Prefetch Gradients: Layer {next_grad_layer_id}")
                            layer.fetch_weights(self.grad_weights_gpu[cur_grad_weight_idx], self.cpu_grad_weights[next_grad_layer_id])
                            self.profiler.range_pop()
                    
                    self.grad_weight_inbound_events[next_grad_layer_id] = self.inbound_stream.record_event()
                    self.grad_weight_inbound_events[k] = None
                    
                    if k_ind - self.n_gpu_grads == 0:
                        self.first_grad_layer_index_for_step = cur_grad_weight_idx
                
                if cur_grad_weight_idx == 0:
                    cur_grad_weight_idx = self.n_gpu_grads - 1
                else:
                    cur_grad_weight_idx -= 1


                ### finish layer
                with self.compute_stream:
                    self.profiler.range_pop()
            
            if self.embed_layer is not None:
                with self.compute_stream:
                    self.profiler.range_push("Embed Backward")

                    if self.zero_grad:
                        for name, tensor in self.embed_gpu["grad_weights"].items():
                            tensor.zero_()
                    
                for j in range(len(chunk_mapping)):
                    chunk = chunk_mapping[j]
                    chunk_metadata = chunk["chunk_metadata"]
                    chunk_token_ids = chunk["chunk_token_ids"]
                    chunk_label_ids = chunk["chunk_label_ids"]

                    with self.compute_stream:
                        self.profiler.range_push(f"Chunk {j}")
                        self.embed_layer.backward(self.transitions_gpu[j], chunk_token_ids, self.embed_gpu["grad_weights"])
                        self.profiler.range_pop()
                
                ## send gradient home (optional duplication), becomes slight bottlenck at end
                ## of last accumlation round when we call synchronize() before returning
                """
                self.outbound_stream.wait_stream(self.compute_stream)
                with self.outbound_stream:
                    self.profiler.range_push("Send Gradients Home: Embed")
                    for name, tensor in self.embed_gpu["grad_weights"].items():
                        self.cpu_embed["grad_weights"][name].copy_(tensor, non_blocking=True)
                    self.profiler.range_pop()
                """

                with self.compute_stream:
                    self.profiler.range_pop()
            

            ### finish bwd
            with self.compute_stream:
                self.profiler.range_pop()

            with self.compute_stream:
                self.profiler.range_pop()

            ### for simplicity wait here (cpu thread blocking) after each round
            self.compute_stream.synchronize()

            ## we have computed gradients that we need to accumulate (fetch back)
            self.zero_grad = False

            round_num += 1

            ### cleanup to mitigate against fragmentation
            ## gc.collect() doesnt save much memory and appears to come at perf cost... worth looking into this deeper
            #gc.collect()
            torch.cuda.empty_cache()

        ## ensure all updated gradients are sent home before returning
        self.compute_stream.synchronize()
        self.outbound_stream.synchronize()
        
        
    

    def step(self, opt_hyperparams, debug=False):

        with self.compute_stream:
            self.profiler.range_push("Optimizer Step")
        
        ## no matter what the next fwd bwd we start we will start with zero gradients
        ## set here in case of failure (nan/inf gradient short causing early return)
        self.zero_grad = True

        if self.embed_layer is not None:
            with self.compute_stream:
                self.profiler.range_push("Embed")
                ret = self.embed_layer.step(self.embed_gpu["weights"], self.embed_gpu["grad_weights"], self.embed_gpu["opt_state"], opt_hyperparams)
                if ret:
                    print(f"Step failed for embed")
                    return -1
                self.profiler.range_pop()

            ## save updated weights to host for safekeeping (optional duplication for faster saving directly from cpu memory)
            """
            self.outbound_stream.wait_stream(self.compute_stream)
            with self.outbound_stream:
                self.profiler.range_push("Save Updated Weights: Embed")
                for name, tensor in self.embed_gpu["weights"].items():
                    self.cpu_embed["weights"][name].copy_(tensor, non_blocking=True)
                self.profiler.range_pop()
                self.profiler.range_push("Save Updated Opt State: Embed")
                for name, tensor in self.embed_gpu["opt_state"].items():
                    self.cpu_embed["opt_state"][name].copy_(tensor, non_blocking=True)
                self.profiler.range_pop()
            """
        
        if self.head_layer is not None:
            with self.compute_stream:
                self.profiler.range_push("Head")
                ret = self.head_layer.step(self.head_gpu["weights"], self.head_gpu["grad_weights"], self.head_gpu["opt_state"], opt_hyperparams)
                if ret:
                    print(f"Step failed for head")
                    return -1
                self.profiler.range_pop()
            
            ## save updated weights to host for safekeeping (optional duplication for faster saving directly from cpu memory)
            """
            self.outbound_stream.wait_stream(self.compute_stream)
            with self.outbound_stream:
                self.profiler.range_push("Save Updated Weights: Head")
                for name, tensor in self.head_gpu["weights"].items():
                    self.cpu_head["weights"][name].copy_(tensor, non_blocking=True)
                self.profiler.range_pop()
                self.profiler.range_push("Save Updated Opt State: Head")
                for name, tensor in self.head_gpu["opt_state"].items():
                    self.cpu_head["opt_state"][name].copy_(tensor, non_blocking=True)
                self.profiler.range_pop()
            """

        with self.compute_stream:
            self.profiler.range_push("Clear GPU Activations")
            self.clear_gpu_activations()
            self.profiler.range_pop()

        with self.compute_stream:
            self.profiler.range_push("Create GPU Opt State")
            self.create_gpu_opt_state()
            self.profiler.range_pop()

        cur_weight_idx = self.first_weight_layer_index_for_step
        cur_grad_idx = self.first_grad_layer_index_for_step
        cur_opt_idx = 0

        ### the first n_gpu_model_layers and n_gpu_grads should be available...
        for k_ind in range(len(self.local_layer_ids)):

            layer_id = self.local_layer_ids[k_ind]
            layer = self.model_layers[layer_id]

            self.compute_stream.wait_event(self.weight_inbound_events[layer_id])
            self.compute_stream.wait_event(self.grad_weight_inbound_events[layer_id])
            self.compute_stream.wait_event(self.opt_inbound_events[layer_id])

            with self.compute_stream:
                self.profiler.range_push(f"Layer {layer_id}")
                ret = layer.step(self.model_weights_gpu[cur_weight_idx], self.grad_weights_gpu[cur_grad_idx], self.opt_weights_gpu[cur_opt_idx], opt_hyperparams)
                if ret:
                    print(f"Step failed for layer {layer_id}")
                    return -1
                self.profiler.range_pop()

            self.outbound_stream.wait_stream(self.compute_stream)
            with self.outbound_stream:
                self.profiler.range_push(f"Save Updated Weights: Layer {layer_id}")
                for name, tensor in self.model_weights_gpu[cur_weight_idx].items():
                    self.cpu_model_weights[layer_id][name].copy_(tensor, non_blocking=True)
                self.profiler.range_pop()
                self.profiler.range_push(f"Save Updated Opt State: Layer {layer_id}")
                for name, tensor in self.opt_weights_gpu[cur_opt_idx].items():
                    self.cpu_opt_weights[layer_id][name].copy_(tensor, non_blocking=True)
                self.profiler.range_pop()

            ## Replace weights/gradients/opt state
            self.inbound_stream.wait_stream(self.outbound_stream)

            if k_ind + self.n_gpu_model_layers < len(self.local_layer_ids):
                next_weight_layer_id = self.local_layer_ids[k_ind + self.n_gpu_model_layers]
                next_weight_layer = self.model_layers[next_weight_layer_id]
                with self.inbound_stream:
                    self.profiler.range_push(f"Inbound Weights: Layer {next_weight_layer_id}")
                    next_weight_layer.fetch_weights(self.model_weights_gpu[cur_weight_idx], self.cpu_model_weights[next_weight_layer_id])
                    self.profiler.range_pop()                   
                self.weight_inbound_events[next_weight_layer_id] = self.inbound_stream.record_event()
                self.weight_inbound_events[layer_id] = None
                    
                
            if k_ind + self.n_gpu_grads < len(self.local_layer_ids):
                next_grad_layer_id = self.local_layer_ids[k_ind + self.n_gpu_grads]
                next_grad_layer = self.model_layers[next_grad_layer_id]
                with self.inbound_stream:
                    self.profiler.range_push(f"Inbound Gradients: Layer {next_grad_layer_id}")
                    next_grad_layer.fetch_weights(self.grad_weights_gpu[cur_grad_idx], self.cpu_grad_weights[next_grad_layer_id])
                    self.profiler.range_pop()
                self.grad_weight_inbound_events[next_grad_layer_id] = self.inbound_stream.record_event()
                self.grad_weight_inbound_events[layer_id] = None

            if k_ind + self.n_gpu_opt_layers < len(self.local_layer_ids):
                next_opt_layer_id = self.local_layer_ids[k_ind + self.n_gpu_opt_layers]
                next_opt_layer = self.model_layers[next_opt_layer_id]
                with self.inbound_stream:
                    self.profiler.range_push(f"Inbound Opt State: Layer {next_opt_layer_id}")
                    next_opt_layer.fetch_weights(self.opt_weights_gpu[cur_opt_idx], self.cpu_opt_weights[next_opt_layer_id])
                    self.profiler.range_pop()
                self.opt_inbound_events[next_opt_layer_id] = self.inbound_stream.record_event()
                self.opt_inbound_events[layer_id] = None

            #self.compute_stream.wait_stream(self.outbound_stream)
            
            cur_weight_idx = (cur_weight_idx + 1) % self.n_gpu_model_layers
            cur_grad_idx = (cur_grad_idx + 1) % self.n_gpu_grads
            cur_opt_idx = (cur_opt_idx + 1) % self.n_gpu_opt_layers

            
        ### Reload early layers to get ready for next fwd_bwd...
        self.inbound_stream.wait_stream(self.compute_stream)
        self.inbound_stream.wait_stream(self.outbound_stream)

        cur_weight_idx = 0

        self.weight_inbound_events.clear()
 
        for i in range(self.n_gpu_model_layers):
            layer_id = self.local_layer_ids[i]
            layer = self.model_layers[layer_id]
            with self.inbound_stream:
                self.profiler.range_push(f"Reload Weights: Layer {layer_id}")
                layer.fetch_weights(self.model_weights_gpu[cur_weight_idx], self.cpu_model_weights[layer_id])
                self.profiler.range_pop()

            self.weight_inbound_events[layer_id] = self.inbound_stream.record_event()

            ## we started at 0, so reloading should fill up the first n_gpu_model_layers
            cur_weight_idx += 1

        ## i think these are overkill... but need to design API's "sync contract"
        self.inbound_stream.synchronize()
        self.compute_stream.synchronize()
        self.outbound_stream.synchronize()


        ## zero out gradients

        
        """
        Wasteful to do this, can incur a lot of overhead particularly in host mem bw bound regimes...
        instead just set device buffer to be 0 on first round of next step
        with self.compute_stream:
            for i in range(len(self.local_layer_ids)):
                layer_id = self.local_layer_ids[i]
                for name, tensor in self.cpu_grad_weights[layer_id].items():
                    tensor.zero_()
        """

        with self.compute_stream:
            self.profiler.range_push("Clear GPU Opt State")
            self.clear_gpu_opt_state()
            self.profiler.range_pop()

        ## recreate activations
        with self.compute_stream:
            self.profiler.range_push("Create GPU Activations")
            self.create_gpu_activations()
            self.profiler.range_pop()

        with self.compute_stream:
            self.profiler.range_pop()


        self.step_count += 1

        return 0

        
