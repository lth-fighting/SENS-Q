"""
Main quantization pipeline (SENS‑Q) as described in Section III‑D.
Command‑line interface for quantizing a trained model.
"""

import os
import sys
import logging
import argparse
import copy
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Local imports
from models.model_architecture import CNNMambaTransformer
from models.dataset import ProteinExpressionDataset
from quantization.quant_layer import QuantLinearLUT, find_layers, find_module, make_quant_lut
from quantization.sensitivity import compute_gradients, remove_outliers
from quantization.clustering import parallel_finetune

logger = logging.getLogger(__name__)

class QuantizationArguments:
    """Container for quantization hyper‑parameters."""
    def __init__(self, model_path, output_dir, data_dir, bit=3,
                 sensitivity=0.05, range=None, balanced=False,
                 num_nonzero_per_thread=10, num_examples=200, batch_size=32):
        self.model_path = model_path
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.bit = bit
        self.sensitivity = sensitivity  # fraction of top‑sensitivity weights
        self.range = range
        self.balanced = balanced
        self.num_nonzero_per_thread = num_nonzero_per_thread
        self.num_examples = num_examples  # calibration set size (paper uses 200)
        self.batch_size = batch_size

def load_model_with_checkpoint(model_path, model_class, model_args):
    """Robust model loading from checkpoint or state_dict."""
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model = model_class(**model_args)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
    if not filtered:
        raise ValueError("No matching parameters found in the checkpoint.")
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model

def quantize_model(args):
    """Run the complete SENS‑Q quantization pipeline."""
    logger.info("="*60)
    logger.info(f"SENS‑Q Quantization: bits={args.bit}, sensitivity={args.sensitivity}")
    logger.info("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_args = {
        'k': 3, 'num_bio_features': 7, 'cnn_out_channels': 64,
        'mamba_hidden_size': 64, 'transformer_heads': 4,
        'transformer_layers': 2, 'fusion_method': 'weighted_sum'
    }

    # Load model
    model = load_model_with_checkpoint(args.model_path, CNNMambaTransformer, model_args)
    model.to('cpu')

    # Prepare calibration data
    train_dataset = ProteinExpressionDataset(os.path.join(args.data_dir, 'train_Ecoli_data.csv'), k=3)
    indices = torch.randperm(len(train_dataset))[:args.num_examples]  # default 200
    subset = Subset(train_dataset, indices)
    calib_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)

    # Compute squared‑gradient sensitivities
    g_model = copy.deepcopy(model)
    for p in g_model.parameters():
        p.requires_grad = True
    g_model = compute_gradients(g_model, calib_loader, device)

    linear_layers = find_layers(model, [nn.Linear])
    linear_layers_grad = find_layers(g_model, [nn.Linear])

    # Determine layers to quantize
    if args.range:
        rng = [int(x) for x in args.range.split(',')]
        layer_indices = list(range(rng[0], rng[1]))
    else:
        layer_indices = list(range(len(linear_layers)))

    model_dict = {name: layer.weight.data.clone() for name, layer in linear_layers.items()}
    grad_dict = {name: linear_layers_grad[name].weight.grad.clone() for name in linear_layers.keys()}

    n_cluster = 2 ** (args.bit - 1)
    quantile = 3.0
    step = quantile / (n_cluster + 1)

    all_cluster_results = {}
    all_outlier_configs = {}

    totals_params = 0
    totals_outliers = 0

    for layer_idx, (layer_name, layer) in enumerate(linear_layers.items()):
        if layer_idx not in layer_indices:
            continue
        logger.info(f"Quantizing layer {layer_idx}: {layer_name}")

        data = {layer_name: layer.weight.data.clone().detach()}
        grad = {layer_name: grad_dict[layer_name].clone().detach()}

        w_np = data[layer_name].cpu().numpy()
        outlier_config = []
        cluster_init = []

        for i in range(w_np.shape[0]):
            row = w_np[i, :]
            mu, std = np.mean(row), np.std(row)
            lo, hi = mu - quantile * std, mu + quantile * std
            outlier_config.append([lo, hi])
            # Non‑uniform initial centroids
            centers = []
            for j in range(1, n_cluster + 1):
                centers.append(mu - step * j * std)
                centers.append(mu + step * j * std)
            cluster_init.append(centers)

        all_cluster_results[layer_name] = cluster_init
        all_outlier_configs[layer_name] = outlier_config

        totals_params += w_np.size
        totals_outliers += int(( (w_np < np.array([lo for lo,_ in outlier_config])) |
                                 (w_np > np.array([hi for _,hi in outlier_config])) ).sum())

        # Remove outliers
        if args.sensitivity > 0:
            _ = remove_outliers(
                model=data,
                sensitivity=args.sensitivity,
                outlier_config={layer_name: outlier_config},
                gradients=grad
            )
        else:
            pass  # only threshold removal if no sensitivity

        # Sensitivity‑weighted clustering
        g = grad[layer_name].float().numpy()
        w = data[layer_name].float().numpy()
        kmeans_res = parallel_finetune(w, g, all_cluster_results, layer_name)

        # Assemble lookup table
        config_per_row = []
        for i, (centers, labels) in enumerate(kmeans_res):
            config_per_row.append([(centers, labels)])

        lookup_table = [config_per_row, None]  # outliers are already embedded in weights

        # Replace layer with QuantLinearLUT
        _, qmod = find_module(model, layer_name, [nn.Linear])
        if qmod is None:
            logger.error(f"Could not find layer {layer_name}")
            continue

        make_quant_lut(model, [layer_name], args.bit,
                       include_sparse=(args.sensitivity > 0),
                       balanced=args.balanced,
                       num_nonzero_per_thread=args.num_nonzero_per_thread)

        # Pack weights into quantized layer
        _, qmod = find_module(model, layer_name, [QuantLinearLUT])
        if qmod is not None:
            qmod.pack2(layer, lookup_table, args.sensitivity > 0,
                       num_nonzero_per_thread=args.num_nonzero_per_thread)

    # Save quantized model
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    torch.save(model.state_dict(), args.output_dir)
    logger.info(f"Quantized model saved to {args.output_dir}")
    return model, totals_params, totals_outliers

def main():
    parser = argparse.ArgumentParser(description='SENS‑Q Post‑Training Quantization')
    parser.add_argument('--model_path', type=str, default='./best_model.pth')
    parser.add_argument('--output_dir', type=str, default='./quantized_model.pth')
    parser.add_argument('--data_dir', type=str, default='./processed_data')
    parser.add_argument('--bit', type=int, default=3, choices=[2,3,4])
    parser.add_argument('--sensitivity', type=float, default=0.05, help='Fraction of top‑sensitivity weights to protect')
    parser.add_argument('--num_examples', type=int, default=200, help='Calibration set size (paper uses 200)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--range', type=str, default=None)
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--num_nonzero_per_thread', type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    quant_args = QuantizationArguments(**vars(args))
    quantize_model(quant_args)

if __name__ == '__main__':
    main()
