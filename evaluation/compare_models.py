"""
Head‑to‑head comparison script between original and quantized models.
Usage: python -m evaluation.compare_models --original_model ... --quantized_model ...
"""

import os
import sys
import json
import logging
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

from models.model_architecture import CNNMambaTransformer
from models.dataset import ProteinExpressionDataset
from evaluation.model_io import load_original_model, build_quantized_model_from_state_dict
from evaluation.metrics import (evaluate_model, compute_metrics,
                                get_model_size_mb, count_parameters,
                                measure_inference_time)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_model', required=True)
    parser.add_argument('--quantized_model', required=True)
    parser.add_argument('--data_dir', default='./processed_data')
    parser.add_argument('--output_dir', default='./comparison_results')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    os.makedirs(args.output_dir, exist_ok=True)

    model_args = {
        'k': 3, 'num_bio_features': 7, 'cnn_out_channels': 64,
        'mamba_hidden_size': 64, 'transformer_heads': 4,
        'transformer_layers': 2, 'fusion_method': 'weighted_sum'
    }

    test_dataset = ProteinExpressionDataset(os.path.join(args.data_dir, 'test_Ecoli_data.csv'), k=3)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load models
    orig_model = load_original_model(args.original_model, model_args, device)
    quant_state = torch.load(args.quantized_model, map_location='cpu')
    if 'model_state_dict' in quant_state:
        quant_state = quant_state['model_state_dict']
    quant_model = build_quantized_model_from_state_dict(CNNMambaTransformer, model_args, quant_state, device)

    # Evaluate
    _, orig_tgt, orig_pred = evaluate_model(orig_model, test_loader, device)
    _, quant_tgt, quant_pred = evaluate_model(quant_model, test_loader, device)
    orig_metrics = compute_metrics(orig_tgt, orig_pred)
    quant_metrics = compute_metrics(quant_tgt, quant_pred)

    orig_size = get_model_size_mb(orig_model)
    quant_size = get_model_size_mb(quant_model)
    orig_params = count_parameters(orig_model) / 1e6
    quant_params = count_parameters(quant_model) / 1e6
    orig_time = measure_inference_time(orig_model, test_loader, device)
    quant_time = measure_inference_time(quant_model, test_loader, device)

    report = {
        'FP32': {'r2': orig_metrics['r2'], 'mae': orig_metrics['mae'],
                 'mse': orig_metrics['mse'], 'params_M': orig_params,
                 'size_MB': orig_size, 'infer_ms': orig_time},
        'Quantized': {'r2': quant_metrics['r2'], 'mae': quant_metrics['mae'],
                      'mse': quant_metrics['mse'], 'params_M': quant_params,
                      'size_MB': quant_size, 'infer_ms': quant_time}
    }
    with open(os.path.join(args.output_dir, 'comparison.json'), 'w') as f:
        json.dump(report, f, indent=2)

    logger.info("Comparison report saved.")

if __name__ == '__main__':
    main()
