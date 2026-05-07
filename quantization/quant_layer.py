"""
QuantLinearLUT and layer‑replacement utilities.
Implements the hardware‑friendly lookup‑table linear module (Section III‑D4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

def find_layers(module, layers=[nn.Linear], name=""):
    """Recursively find all layers of specified types."""
    if type(module) in layers:
        return {name: module}
    res = {}
    for child_name, child in module.named_children():
        full_name = name + "." + child_name if name else child_name
        res.update(find_layers(child, layers, full_name))
    return res

def find_module(module, module_name, layers=[nn.Linear], name=""):
    """Locate a specific sub‑module by name (dot‑separated path)."""
    if type(module) in layers and name.find(module_name) != -1:
        return True, module
    for child_name, child in module.named_children():
        full_name = name + "." + child_name if name else child_name
        is_find, mod = find_module(child, module_name, layers, full_name)
        if is_find:
            return True, mod
    return False, None

class QuantLinearLUT(nn.Module):
    """
    Quantized linear layer using a per‑row lookup table (LUT) and optional sparse
    full‑precision outlier storage (CSR format). Only 2, 3, 4 bits supported.
    """
    def __init__(self, bits, infeatures, outfeatures, bias,
                 include_sparse=False, numvals=0, topX=0,
                 balanced=False, num_nonzero_per_thread=10):
        super().__init__()
        if bits not in [2, 3, 4]:
            raise NotImplementedError("Only 2, 3, 4 bit quantization supported")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits

        self.register_buffer('qweight', torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32))
        self.register_buffer('lookup_table', torch.zeros((outfeatures, 2**bits), dtype=torch.float32))

        self.include_bias = bias
        if bias:
            self.register_buffer('bias', torch.zeros(outfeatures))
        else:
            self.bias = None

        self.include_sparse = include_sparse
        self.numvals = numvals
        self.topX = topX
        if numvals > 0:
            self.register_buffer('rows', torch.zeros(outfeatures + 1, dtype=torch.int32))
            self.register_buffer('cols', torch.zeros(numvals, dtype=torch.int32))
            self.register_buffer('vals', torch.zeros(numvals, dtype=torch.float32))
        if topX > 0:
            self.register_buffer('full_rows', torch.zeros((infeatures, topX), dtype=torch.float32))
            self.register_buffer('full_row_indices', torch.zeros(topX, dtype=torch.int32))
        self.balanced = balanced

    @property
    def weight(self):
        """Return de‑quantized weight matrix (out_features x in_features)."""
        if hasattr(self, 'dequant_weight'):
            return self.dequant_weight.t()
        return torch.zeros((self.outfeatures, self.infeatures), dtype=torch.float32)

    @property
    def in_features(self):
        return self.infeatures

    @property
    def out_features(self):
        return self.outfeatures

    def pack2(self, linear, lookup_table, include_sparse, num_nonzero_per_thread=-1):
        """Pack a standard nn.Linear into the quantized LUT format."""
        if self.include_bias and linear.bias is not None:
            self.bias = linear.bias.clone()

        lut, outliers = lookup_table
        intweight = linear.weight.data.clone()
        return_weight = torch.zeros_like(intweight)
        indices_tensor = torch.zeros_like(intweight, dtype=torch.int32)

        num_channels = len(lut)
        for ch in range(num_channels):
            centroid, indices = lut[ch][0]  # centroid is array, indices is array
            indices_tensor[ch] = torch.from_numpy(indices)
            self.lookup_table[ch] = torch.from_numpy(centroid)
            return_weight[ch] = torch.tensor([centroid[i] for i in indices])

            if include_sparse and outliers is not None:
                zero_mapping = self._round_to_nearest_pole(torch.zeros(1), centroid)
                nonzero_vals = torch.nonzero(outliers[ch])
                return_weight[ch][nonzero_vals] = outliers[ch][nonzero_vals]
                outliers_channel = outliers[ch]
                outliers_channel[nonzero_vals] -= zero_mapping.to(outliers_channel.device)
                outliers[ch] = outliers_channel

        if include_sparse and outliers is not None:
            sparse = outliers.to_sparse(layout=torch.sparse_csr)
            self.register_buffer('rows', sparse.crow_indices().to(torch.int32))
            self.register_buffer('cols', sparse.col_indices().to(torch.int32))
            self.register_buffer('vals', sparse.values().to(torch.float32))

        self.register_buffer('indices', indices_tensor.to(torch.int32))
        if linear.weight.shape != return_weight.shape:
            raise ValueError(f"Shape mismatch: {linear.weight.shape} vs {return_weight.shape}")
        self.register_buffer('dequant_weight', return_weight.t().contiguous())

        # Pack indices into compact qweight (for compatibility)
        intw = indices_tensor.t().contiguous().numpy().astype(np.uint32)
        qw = np.zeros((intw.shape[0] // 32 * self.bits, intw.shape[1]), dtype=np.uint32)
        i = 0
        row = 0
        while row < qw.shape[0]:
            if self.bits in [2, 4]:
                for j in range(i, i + 32 // self.bits):
                    qw[row] |= intw[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qw[row] |= intw[j] << (3 * (j - i))
                i += 10
                qw[row] |= intw[i] << 30
                row += 1
                qw[row] |= (intw[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qw[row] |= intw[j] << (3 * (j - i) + 1)
                i += 10
                qw[row] |= intw[i] << 31
                row += 1
                qw[row] |= (intw[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qw[row] |= intw[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError
        self.qweight = torch.from_numpy(qw.astype(np.int32))
        return return_weight

    @staticmethod
    def _round_to_nearest_pole(w, poles):
        """Return the nearest pole for each weight element."""
        diff = torch.stack([(w - c).abs() for c in poles])
        idx = diff.argmin(0)
        aug = sum((idx == i) * c for i, c in enumerate(poles))
        return aug

    def forward(self, x):
        if self.bias is not None:
            return F.linear(x, self.weight, self.bias)
        return F.linear(x, self.weight)

    def dequantize(self):
        return self.dequant_weight.t()

    def decode_qweight(self):
        return self.indices.t()

def make_quant_lut(module, names, bits, name="", include_sparse=False,
                   numvals=None, topX=0, balanced=False, num_nonzero_per_thread=10):
    """Replace standard nn.Linear layers with QuantLinearLUT."""
    if isinstance(module, QuantLinearLUT):
        return
    # Check attributes
    for attr in dir(module):
        try:
            tmp = getattr(module, attr)
            full_name = name + "." + attr if name else attr
            if full_name in names and isinstance(tmp, nn.Linear):
                num = numvals.get(full_name, 0) if numvals else 0
                setattr(module, attr, QuantLinearLUT(
                    bits, tmp.in_features, tmp.out_features, tmp.bias is not None,
                    include_sparse=include_sparse, numvals=num, topX=topX,
                    balanced=balanced, num_nonzero_per_thread=num_nonzero_per_thread
                ))
        except:
            pass
    # Also check _modules
    for child_name, child in module._modules.items():
        full_name = name + "." + child_name if name else child_name
        if full_name in names and isinstance(child, nn.Linear):
            num = numvals.get(full_name, 0) if numvals else 0
            module._modules[child_name] = QuantLinearLUT(
                bits, child.in_features, child.out_features, child.bias is not None,
                include_sparse=include_sparse, numvals=num, topX=topX,
                balanced=balanced, num_nonzero_per_thread=num_nonzero_per_thread
            )
    # Recursion
    for child_name, child in module.named_children():
        make_quant_lut(child, names, bits,
                       name + "." + child_name if name else child_name,
                       include_sparse, numvals, topX, balanced, num_nonzero_per_thread)
