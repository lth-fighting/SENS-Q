"""
Protein expression dataset with k‑mer encoding and biophysical feature normalization.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from itertools import product

# ---------- DNA k‑mer encoding ----------
def DNA_kmer_onehot_encode(sequence: str, k: int = 3) -> np.ndarray:
    """Convert DNA sequence into a one‑hot matrix of overlapping k‑mers."""
    bases = ['A', 'T', 'C', 'G']
    all_kmers = [''.join(p) for p in product(bases, repeat=k)]
    kmer_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
    kmer_to_char = {kmer: kmer_chars[i] for i, kmer in enumerate(all_kmers[:len(kmer_chars)])}

    kmer_seq = []
    for i in range(0, len(sequence) - k + 1, 1):
        kmer = sequence[i:i+k].upper()
        kmer_seq.append(kmer_to_char.get(kmer, 'N'))

    char_mapping = {}
    for i, char in enumerate(kmer_chars[:len(kmer_to_char)]):
        one_hot = [0] * len(kmer_to_char)
        one_hot[i] = 1
        char_mapping[char] = one_hot
    char_mapping['N'] = [-1] * len(kmer_to_char)

    encoded = np.array([char_mapping.get(c, [-1]*len(kmer_to_char)) for c in kmer_seq])
    return encoded

# ---------- Dataset Class ----------
class ProteinExpressionDataset(Dataset):
    """
    PyTorch Dataset for the E. coli ribosome stalling prediction task.
    Returns:
        sequence_tensor: (num_channels, seq_len) one‑hot encoded k‑mer matrix
        features: (num_bio_features,) standardized biophysical features
        target: normalized cdsBottleneckRelativeStrength
    """
    BIOPHYSICAL_FEATURES = [
        'cdsCAI', 'utrCdsStructureMFE', 'fivepCdsStructureMFE',
        'threepCdsStructureMFE', 'cdsBottleneckPosition',
        'cdsNucleotideContentAT', 'cdsHydropathyIndex'
    ]

    def __init__(self, csv_file: str, target: str = 'Protein',
                 encoding_method: str = 'kmer_onehot',
                 use_biophysical_features: bool = True, k: int = 3):
        self.data = pd.read_csv(csv_file)
        self.sequences = self.data['Sequence'].values
        self.k = k
        self.use_biophysical = use_biophysical_features

        if use_biophysical_features:
            bio_data = self.data[self.BIOPHYSICAL_FEATURES].values
            scaler = StandardScaler()
            self.features = scaler.fit_transform(bio_data)

        # Normalize target as described in the paper
        target_raw = self.data[target].values
        self.target = (target_raw - target_raw.mean()) / target_raw.std()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_encoded = DNA_kmer_onehot_encode(seq, self.k)
        seq_tensor = torch.tensor(seq_encoded, dtype=torch.float32).transpose(0, 1)

        target = torch.tensor(self.target[idx], dtype=torch.float32)
        features = torch.tensor(self.features[idx], dtype=torch.float32) if self.use_biophysical else torch.zeros(7)

        return seq_tensor, features, target
