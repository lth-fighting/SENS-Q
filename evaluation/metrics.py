"""
Evaluation metrics and benchmarking functions.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate_model(model, dataloader, device, criterion=nn.MSELoss()):
    """Return loss, targets, predictions."""
    model.eval()
    total_loss = 0.0
    targets, preds = [], []
    with torch.no_grad():
        for seq, bio, target in dataloader:
            seq, bio, target = seq.to(device), bio.to(device), target.to(device)
            out = model(seq, bio)
            loss = criterion(out, target)
            total_loss += loss.item() * seq.size(0)
            targets.extend(target.cpu().numpy())
            preds.extend(out.cpu().numpy())
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, np.array(targets), np.array(preds)

def compute_metrics(targets, predictions):
    """Compute R², MAE, MSE."""
    return {
        'r2': r2_score(targets, predictions),
        'mae': mean_absolute_error(targets, predictions),
        'mse': mean_squared_error(targets, predictions)
    }

def get_model_size_mb(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buf_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buf_size) / (1024 ** 2)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def measure_inference_time(model, dataloader, device, num_batches=50):
    """Average batch inference time in milliseconds."""
    model.eval()
    times = []
    with torch.no_grad():
        for i, (seq, bio, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            seq, bio = seq.to(device), bio.to(device)
            if device.type == 'cuda':
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                starter.record()
                _ = model(seq, bio)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))
            else:
                start = time.time()
                _ = model(seq, bio)
                end = time.time()
                times.append((end - start) * 1000)
    return np.mean(times)
