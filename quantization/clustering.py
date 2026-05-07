"""
Sensitivity‑weighted K‑means fine‑tuning (Section III‑D3).
Uses squared gradient as weight to pull centroids toward high‑impact regions.
"""

import numpy as np
from multiprocessing import Pool, cpu_count

def finetune_center(args):
    """
    Fine‑tune one row’s cluster centers using sensitivity‑weighted distances.
    args: (i, weights, grads, cluster_results, module_name)
    """
    i, weights, grads, cluster_results, module_name = args
    centers = cluster_results[module_name][i]
    grads = grads / (grads.max() - grads.min() + 1e-12)
    mask = weights != 0
    weights = weights * mask
    grads = grads * mask

    labels = np.argmin(np.abs(weights[:, None] - centers), axis=1)
    best_loss = (np.abs(weights - np.take(centers, labels)) * grads).sum()
    eps = 1e-6
    patience = 0
    max_patience = 20

    while patience <= max_patience:
        new_centers = np.empty_like(centers)
        labels = np.argmin(np.abs(weights[:, None] - centers), axis=1)
        for j in range(len(centers)):
            center_mask = labels == j
            if center_mask.sum() > 0:
                new_centers[j] = (weights * center_mask * grads).sum() / (center_mask * grads).sum()
            else:
                new_centers[j] = centers[j]
        new_labels = np.argmin(np.abs(weights[:, None] - new_centers), axis=1)
        new_weights = np.take(new_centers, new_labels)
        loss = (np.square(weights - new_weights) * grads).sum()
        if loss < best_loss - eps:
            best_loss = loss
            centers = new_centers
            patience = 0
        else:
            patience += 1
    return (centers, labels)

def parallel_finetune(module_weight, g, cluster_results, module_name):
    """Parallel fine‑tuning across all output channels."""
    args_list = []
    for i in range(module_weight.shape[0]):
        w = module_weight[i, :]
        gw = g[i, :]
        if np.sum(gw) == 0:
            gw = np.ones_like(gw)
        args_list.append((i, w, gw, cluster_results, module_name))
    with Pool(cpu_count()) as pool:
        results = pool.map(finetune_center, args_list)
    return results
