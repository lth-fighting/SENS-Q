"""
Gradient‑based sensitivity computation and outlier removal (Sections III‑D1, D2).
"""

import torch
import torch.nn as nn
import copy
import logging
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

def square_grad_hook(grad):
    """Hook: replace gradient with its square to compute sensitivity."""
    return grad.pow(2)

def compute_gradients(model, dataloader, device):
    """
    Forward‑backward pass on calibration data, storing squared gradients
    as the sensitivity measure (Eq. 4).
    """
    logger.info("Computing squared‑gradient sensitivities...")
    model.train()
    model.to(device)
    hooks = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            hooks.append(param.register_hook(square_grad_hook))

    criterion = nn.MSELoss()
    model.zero_grad()
    for batch_idx, (seq, bio, target) in enumerate(tqdm(dataloader, desc="Gradient calc")):
        if batch_idx >= len(dataloader):
            break
        seq, bio, target = seq.to(device), bio.to(device), target.to(device)
        out = model(seq, bio)
        loss = criterion(out, target)
        loss.backward()

    for hook in hooks:
        hook.remove()
    model.eval()
    model.cpu()
    logger.info("Sensitivity calculation complete.")
    return model

def remove_outliers_by_sensitivity(model, gradients, sensitivity):
    """
    Retain the top (sensitivity * 100)% of weights as full‑precision outliers.
    (sensitivity is a fraction, e.g., 0.05 for top 5%)
    """
    module_names = list(model.keys())
    outlier_weights_list = []
    total_outliers = 0
    total_weights = 0

    def _process(weight, grad_w):
        num_outliers = int(grad_w.numel() * sensitivity)  # FIXED from paper: fraction
        if num_outliers == 0:
            return weight, torch.zeros_like(weight), 0, weight.numel()
        thres = grad_w.reshape(-1).topk(num_outliers).values[-1]
        mask = grad_w > thres
        outlier = weight * mask
        weight = weight * ~mask
        return weight.to(weight.dtype), outlier, mask.sum().item(), mask.numel()

    for name in module_names:
        w = model[name].to(torch.float)
        gw = gradients[name].to(torch.float)
        new_w, out_w, n_out, n_total = _process(w, gw)
        model[name] = new_w
        total_outliers += n_out
        total_weights += n_total
        outlier_weights_list.append(out_w)

    logger.info(f"Outlier fraction (sensitivity‑based): {total_outliers/total_weights*100:.2f}%")
    return [outlier_weights_list]

def remove_outliers_by_threshold(model, outlier_config, outlier_weights=None):
    """Remove statistical outliers (±3σ threshold)."""
    module_names = list(model.keys())
    if outlier_weights is None:
        outlier_weights = [[torch.zeros(1) for _ in module_names]]

    total_outliers = 0
    total_weights = 0

    def _process_row(row, thres):
        mask = torch.logical_or(row <= thres[0], row >= thres[1])
        outlier = row * mask
        row = row * ~mask
        return row.to(row.dtype), outlier, mask.sum().item(), row.numel()

    for i, name in enumerate(module_names):
        thres = outlier_config[name]
        w = model[name].to(torch.float)
        new_rows = []
        out_rows = []
        for r, t in zip(w, thres):
            nr, orr, nout, ntot = _process_row(r, t)
            new_rows.append(nr)
            out_rows.append(orr)
            total_outliers += nout
            total_weights += ntot
        model[name] = torch.stack(new_rows)
        outlier_weights[0][i] = torch.stack(out_rows) + outlier_weights[0][i]

    logger.info(f"Outlier fraction (threshold): {total_outliers/total_weights*100:.2f}%")
    return outlier_weights

def remove_outliers(model, sensitivity, outlier_config, gradients=None):
    """Combined outlier removal: sensitivity‑guided + threshold‑based."""
    if sensitivity > 0:
        logger.info("Removing sensitivity‑based outliers")
        outlier_w = remove_outliers_by_sensitivity(model, gradients, sensitivity)
    else:
        outlier_w = None

    if outlier_config is not None:
        logger.info("Removing threshold‑based outliers")
        outlier_w = remove_outliers_by_threshold(model, outlier_config, outlier_w)

    return outlier_w
