#!/usr/bin/env python3
"""
Experiment 1: Performance and efficiency across bit‑widths (2,3,4) vs FP32.
Generates Table I style results.
"""

import os, sys, time, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_architecture import CNNMambaTransformer
from models.dataset import ProteinExpressionDataset
from evaluation.model_io import load_original_model, build_quantized_model_from_state_dict
from evaluation.metrics import (evaluate_model, compute_metrics,
                                get_model_size_mb, count_parameters,
                                measure_inference_time)
from quantization.quantize_main import quantize_model, QuantizationArguments
from utils.plotting import set_science_style

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = './processed_data'
OUTPUT_DIR = './exp1_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_ARGS = {
    'k': 3, 'num_bio_features': 7, 'cnn_out_channels': 64,
    'mamba_hidden_size': 64, 'transformer_heads': 4,
    'transformer_layers': 2, 'fusion_method': 'weighted_sum'
}

def main():
    set_science_style()
    test_dataset = ProteinExpressionDataset(os.path.join(DATA_DIR, 'test_Ecoli_data.csv'), k=3)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # FP32 baseline
    orig_model = load_original_model('./best_model.pth', MODEL_ARGS, DEVICE)
    _, orig_tgt, orig_pred = evaluate_model(orig_model, test_loader, DEVICE)
    orig_r2 = r2_score(orig_tgt, orig_pred)
    fp32_mem = get_model_size_mb(orig_model)
    fp32_time = measure_inference_time(orig_model, test_loader, DEVICE)

    results = []
    for bits in [2, 3, 4]:
        print(f"Quantizing {bits}-bit...")
        temp_path = f'./temp_quant_{bits}.pth'
        args = QuantizationArguments(
            model_path='./best_model.pth', output_dir=temp_path, data_dir=DATA_DIR,
            bit=bits, sensitivity=0.05, num_examples=200, batch_size=32
        )
        quant_model, _, _ = quantize_model(args)
        quant_model.to(DEVICE)

        _, q_tgt, q_pred = evaluate_model(quant_model, test_loader, DEVICE)
        r2 = r2_score(q_tgt, q_pred)
        mae = np.mean(np.abs(q_tgt - q_pred))
        mse = np.mean((q_tgt - q_pred)**2)
        mem = get_model_size_mb(quant_model)
        inf = measure_inference_time(quant_model, test_loader, DEVICE)
        results.append({
            'bits': bits, 'r2': r2, 'mae': mae, 'mse': mse,
            'mem_MB': mem, 'infer_ms': inf
        })
        os.remove(temp_path)

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'performance.csv'), index=False)

    # Quick plot
    fig, ax = plt.subplots()
    bits = [2,3,4]
    ax.plot(bits, [res['r2'] for res in results], 'o-', label='Quantized')
    ax.axhline(orig_r2, color='r', linestyle='--', label='FP32')
    ax.set_xlabel('Bits')
    ax.set_ylabel('R²')
    ax.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'r2_vs_bits.png'))
    plt.close()

if __name__ == '__main__':
    main()
