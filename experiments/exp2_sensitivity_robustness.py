#!/usr/bin/env python3
"""
Experiment 2: Sensitivity threshold robustness and calibration size effect.
"""

import os, sys, json
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
from evaluation.metrics import evaluate_model
from quantization.quantize_main import quantize_model, QuantizationArguments
from utils.plotting import set_science_style

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = './processed_data'
OUTPUT_DIR = './exp2_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_ARGS = {
    'k': 3, 'num_bio_features': 7, 'cnn_out_channels': 64,
    'mamba_hidden_size': 64, 'transformer_heads': 4,
    'transformer_layers': 2, 'fusion_method': 'weighted_sum'
}

def quantize_once(bit, sensitivity, num_examples, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    temp_path = f'./temp_{bit}_{seed}.pth'
    args = QuantizationArguments(
        model_path='./best_model.pth', output_dir=temp_path, data_dir=DATA_DIR,
        bit=bit, sensitivity=sensitivity, num_examples=num_examples, batch_size=32
    )
    quant_model, _, _ = quantize_model(args)
    quant_model.to(DEVICE)
    test_dataset = ProteinExpressionDataset(os.path.join(DATA_DIR, 'test_Ecoli_data.csv'), k=3)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    _, tgt, pred = evaluate_model(quant_model, test_loader, DEVICE)
    r2 = r2_score(tgt, pred)
    os.remove(temp_path)
    return r2

def main():
    set_science_style()
    # Sensitivity thresholds
    thresholds = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]
    r2_vals = []
    for tau in thresholds:
        r2 = quantize_once(bit=3, sensitivity=tau, num_examples=200, seed=42)
        r2_vals.append(r2)
    df = pd.DataFrame({'tau': thresholds, 'r2': r2_vals})
    df.to_csv(os.path.join(OUTPUT_DIR, 'sensitivity_curve.csv'), index=False)

    # Calibration size
    sizes = [50, 100, 200, 400]
    r2_sizes = []
    for sz in sizes:
        r2 = quantize_once(bit=3, sensitivity=0.05, num_examples=sz, seed=42)
        r2_sizes.append(r2)
    pd.DataFrame({'size': sizes, 'r2': r2_sizes}).to_csv(
        os.path.join(OUTPUT_DIR, 'calibration_size.csv'), index=False)

    # Multi‑run stability
    seeds = list(range(10))
    multi_r2 = []
    for s in seeds:
        r2 = quantize_once(bit=3, sensitivity=0.05, num_examples=200, seed=s)
        multi_r2.append(r2)
    stability = pd.DataFrame({'run': range(1,11), 'r2': multi_r2})
    stability.to_csv(os.path.join(OUTPUT_DIR, 'stability.csv'), index=False)
    print(f"Multi‑run mean R²: {np.mean(multi_r2):.4f} ± {np.std(multi_r2):.4f}")

if __name__ == '__main__':
    main()
