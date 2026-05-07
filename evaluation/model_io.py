"""
Model loading utilities for original and quantized models.
"""

import logging
import torch
from models.model_architecture import CNNMambaTransformer
from quantization.quant_layer import QuantLinearLUT, find_layers

logger = logging.getLogger(__name__)

def load_original_model(model_path, model_args, device):
    """Load un‑quantized FP32 model (from checkpoint or state_dict)."""
    logger.info("Loading original FP32 model...")
    model = CNNMambaTransformer(**model_args)
    state = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    elif 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model

def build_quantized_model_from_state_dict(original_model_class, model_args, quant_state_dict, device):
    """
    Reconstruct a quantized model from a saved state_dict containing QuantLinearLUT parameters.
    Automatically replaces standard nn.Linear layers with QuantLinearLUT when lookup_table is present.
    """
    logger.info("Reconstructing quantized model architecture...")
    model = original_model_class(**model_args)
    linear_layers = find_layers(model, [torch.nn.Linear])

    quant_layer_names = set()
    for key in quant_state_dict.keys():
        if key.endswith('.lookup_table'):
            quant_layer_names.add(key[:-len('.lookup_table')])

    for name, layer in linear_layers.items():
        if name in quant_layer_names:
            in_f, out_f = layer.in_features, layer.out_features
            bias = layer.bias is not None
            lookup = quant_state_dict[f"{name}.lookup_table"]
            bits = int(np.log2(lookup.shape[1]))
            qlayer = QuantLinearLUT(bits, in_f, out_f, bias, include_sparse=False)
            parent = model
            *path, leaf = name.split('.')
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, leaf, qlayer)

    missing, unexpected = model.load_state_dict(quant_state_dict, strict=False)
    if unexpected:
        # Register missing buffers (e.g., dequant_weight, indices)
        from collections import defaultdict
        module_bufs = defaultdict(list)
        for key in unexpected:
            *parts, buf_name = key.split('.')
            mod_name = '.'.join(parts)
            module_bufs[mod_name].append((buf_name, key))
        for mod_name, bufs in module_bufs.items():
            mod = model
            for p in mod_name.split('.'):
                mod = getattr(mod, p)
            for buf_name, full_key in bufs:
                tensor = quant_state_dict[full_key]
                if hasattr(mod, buf_name):
                    getattr(mod, buf_name).copy_(tensor)
                else:
                    mod.register_buffer(buf_name, tensor)

    model.to(device)
    model.eval()
    return model
