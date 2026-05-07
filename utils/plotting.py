"""
Common plotting style for IEEE/ACM publications.
"""

import matplotlib.pyplot as plt
import seaborn as sns

def set_science_style():
    sns.set_style("whitegrid")
    sns.set_context("paper")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'axes.linewidth': 1.0,
    })
