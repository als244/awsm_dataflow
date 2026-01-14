
from .dense_layer import TransformerLayer
from .moe_layer import TransformerMoELayer


def get_transformer_saved_act_sizes(model_dims, num_tokens):
    if model_dims["num_routed_experts"] > 0:
        return get_moe_saved_act_sizes(model_dims, num_tokens)
    else:
        return get_dense_saved_act_sizes(model_dims, num_tokens)

def get_moe_saved_act_sizes(model_dims, num_tokens):
    saved_act_sizes = {}
    
    ## create obj to then query get_act_slot_size function
    moe_layer = TransformerMoELayer(0, model_dims, {})

    max_saved_act_level = moe_layer.max_saved_activations_level

    for i in range(max_saved_act_level + 1):
        saved_act_sizes[i] = moe_layer.get_act_slot_size(num_tokens, saved_level=i)
    return saved_act_sizes

def get_dense_saved_act_sizes(model_dims, num_tokens):
    saved_act_sizes = {}
    dense_layer = TransformerLayer(0, model_dims, {})
    max_saved_act_level = dense_layer.max_saved_activations_level
    for i in range(max_saved_act_level + 1):
        saved_act_sizes[i] = dense_layer.get_act_slot_size(num_tokens, saved_level=i)
    return saved_act_sizes
    

