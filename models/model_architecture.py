"""
SENS‑Q Model Architecture

Implements the hybrid deep learning model for ribosome stalling strength prediction
as described in:
"SENS-Q: Sensitivity-Guided Non-Uniform Quantization for Efficient and
 Biologically Faithful Prediction of Ribosome Stalling with Hybrid Deep Learning Models"
 (IEEE/ACM TCBB)
 
Authors: Tianhui Li, Huiping Liu, Weiliang Zeng
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------- Positional Encodings ----------
class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encodings for the Transformer branch."""
    def __init__(self, d_model: int, max_len: int = 94):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.position_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.xavier_uniform_(self.position_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max length {self.max_len}")
        return x + self.position_embedding[:, :seq_len, :]

# ---------- Multi‑Scale CNN ----------
class MultiScaleCNN(nn.Module):
    """Multi‑scale 1D CNN for local motif extraction (kernel sizes 3,5,7)."""
    def __init__(self, in_channels: int, cnn_out_channels: int = 64,
                 kernel_sizes: list = [3, 5, 7], dropout_rate: float = 0.2):
        super().__init__()
        self.num_kernels = len(kernel_sizes)
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, cnn_out_channels, k, padding=k//2),
                nn.BatchNorm1d(cnn_out_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for k in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outs = [branch(x) for branch in self.conv_branches]
        return torch.cat(branch_outs, dim=1)

# ---------- MambaPlus Blocks ----------
class MambaPlusBlock(nn.Module):
    """
    MambaPlus block as defined in Equation (1) of the paper:
    y = SSM(SiLU(Conv1d(x))) ⊗ SiLU(σ(g(x))) + SiLU(Conv1d(x)) ⊗ (1-σ(g(x)))
    """
    def __init__(self, d_model: int, d_state: int = 32, d_conv: int = 4,
                 expand: int = 2, conv_kernel_size: int = 3):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand

        # Branch 1: convolution + SSM
        self.conv1d = nn.Conv1d(
            in_channels=d_model, out_channels=self.d_inner,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2, groups=d_model
        )
        self.ssm = Mamba(d_model=self.d_inner, d_state=d_state, d_conv=d_conv, expand=1)

        # Branch 2: gating
        self.gate_linear = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(self.d_inner, d_model)

        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(self.d_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        # Convolution + SiLU
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.silu(x_conv)
        x_conv = self.layer_norm2(x_conv)

        # SSM
        ssm_out = self.ssm(x_conv)

        # Gating
        gate = self.sigmoid(self.gate_linear(residual))
        forget_gate = 1 - gate
        # Expand gate to match d_inner
        gate_exp = gate.unsqueeze(-1).repeat(1, 1, self.d_inner // self.d_model)
        gate_exp = gate_exp.view(gate.size(0), gate.size(1), self.d_inner)
        forget_exp = forget_gate.unsqueeze(-1).repeat(1, 1, self.d_inner // self.d_model)
        forget_exp = forget_exp.view(forget_gate.size(0), forget_gate.size(1), self.d_inner)

        # Mamba+ core
        combined = ssm_out * self.silu(gate_exp) + x_conv * forget_exp
        out = self.output_proj(combined)
        out = self.layer_norm1(out + residual)
        return out

class BiMambaPlusEncoder(nn.Module):
    """
    Bidirectional MambaPlus encoder: forward + backward with FFN.
    """
    def __init__(self, d_model: int, d_state: int = 32, d_conv: int = 4,
                 expand: int = 2, ff_dim=None, dropout_rate: float = 0.1, num_layers: int = 1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        ff_dim = ff_dim or d_model * 4

        self.forward_mamba = MambaPlusBlock(d_model, d_state, d_conv, expand)
        self.backward_mamba = MambaPlusBlock(d_model, d_state, d_conv, expand)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout_rate)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd = self.forward_mamba(x)
        bwd_in = torch.flip(x, [1])
        bwd = self.backward_mamba(bwd_in)
        bwd = torch.flip(bwd, [1])
        combined = fwd + bwd

        # residual + FFN
        x_res = self.norm1(combined + x)
        x_res = self.dropout(x_res)
        ffn_out = self.ffn(x_res)
        return self.norm2(ffn_out + x_res)

# ---------- Transformer Branch ----------
class SequenceTransformerBlock(nn.Module):
    """Single Transformer encoder block with multi‑head attention."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout_rate, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src2, _ = self.attention(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

class SequenceTransformer(nn.Module):
    """Stacked Transformer encoder."""
    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int = 2048, dropout_rate: float = 0.1, max_len: int = 94):
        super().__init__()
        self.pos_encoder = LearnablePositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            SequenceTransformerBlock(d_model, nhead, dim_feedforward, dropout_rate)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoder(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        return x

# ---------- Feature Fusion ----------
class FeatureFusion(nn.Module):
    """Learnable weighted sum of Mamba and Transformer features (Eq. 2)."""
    def __init__(self, d_model: int, fusion_method: str = 'weighted_sum'):
        super().__init__()
        self.method = fusion_method
        if fusion_method == 'weighted_sum':
            self.mamba_weight = nn.Parameter(torch.tensor(0.5))
            self.transformer_weight = nn.Parameter(torch.tensor(0.5))
        elif fusion_method == 'concat':
            self.fusion_proj = nn.Linear(d_model * 2, d_model)
        elif fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(d_model * 2, 4, batch_first=True)
            self.fusion_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, mamba_feat, transformer_feat):
        if self.method == 'weighted_sum':
            alpha = torch.sigmoid(self.mamba_weight)
            beta = torch.sigmoid(self.transformer_weight)
            total = alpha + beta
            return (alpha / total) * mamba_feat + (beta / total) * transformer_feat
        elif self.method == 'concat':
            fused = torch.cat([mamba_feat, transformer_feat], dim=-1)
            return self.fusion_proj(fused)
        elif self.method == 'attention':
            combined = torch.cat([mamba_feat, transformer_feat], dim=-1)
            attented, _ = self.attention(combined, combined, combined)
            return self.fusion_proj(attented)
        return mamba_feat  # fallback

# ---------- Full Hybrid Model ----------
class CNNMambaTransformer(nn.Module):
    """
    Complete model: Multi‑scale CNN → Mamba+ / Transformer → fusion →
    attention pooling → biophysical features → regression head.
    """
    def __init__(self, k: int = 3, num_bio_features: int = 7,
                 cnn_out_channels: int = 64, mamba_hidden_size: int = 64,
                 transformer_heads: int = 4, transformer_layers: int = 2,
                 fc1_size: int = 128, fc2_size: int = 64,
                 dropout_rate: float = 0.15, fusion_method: str = 'weighted_sum'):
        super().__init__()
        in_channels = 4 ** k
        self.multi_scale_cnn = MultiScaleCNN(in_channels, cnn_out_channels)
        cnn_output_size = cnn_out_channels * 3  # 3 kernels

        self.mamba_branch = nn.Sequential(
            BiMambaPlusEncoder(cnn_output_size, d_state=32, d_conv=4, expand=2,
                               dropout_rate=dropout_rate, num_layers=1),
            nn.LayerNorm(cnn_output_size)
        )
        self.transformer_branch = nn.Sequential(
            SequenceTransformer(cnn_output_size, transformer_heads, transformer_layers,
                                dropout_rate=dropout_rate),
            nn.LayerNorm(cnn_output_size)
        )
        self.feature_fusion = FeatureFusion(cnn_output_size, fusion_method)
        self.attention_pooling = nn.Sequential(
            nn.Linear(cnn_output_size, 1),
            nn.Softmax(dim=1)
        )

        combined_size = cnn_output_size + num_bio_features
        self.fc = nn.Sequential(
            nn.Linear(combined_size, fc1_size),
            nn.LayerNorm(fc1_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(fc1_size, fc2_size),
            nn.LayerNorm(fc2_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(fc2_size, 1)
        )

    def forward(self, seq_input: torch.Tensor, biophysical_features: torch.Tensor) -> torch.Tensor:
        cnn_out = self.multi_scale_cnn(seq_input)
        cnn_out = cnn_out.transpose(1, 2)  # (B, L, C)

        mamba_out = self.mamba_branch(cnn_out)
        transformer_out = self.transformer_branch(cnn_out)

        fused = self.feature_fusion(mamba_out, transformer_out)
        attn_weights = self.attention_pooling(fused)
        pooled = torch.sum(attn_weights * fused, dim=1)

        combined = torch.cat([pooled, biophysical_features], dim=-1)
        return self.fc(combined).squeeze(-1)
