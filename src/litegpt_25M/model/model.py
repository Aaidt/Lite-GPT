import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from typing import Tuple
from omegaconf import OmegaConf

# 1. layernorm ⇒ RMSnorm
# 2. gelu ⇒ swiglu
# 3. GQA
# 4. learned ⇒ sinosoidal ⇒ RoPE
# 5. causal ⇒ flash attn

model_cfg = OmegaConf.load("./configs/model/LiteGPT-25M.yaml")
n_vocab = model_cfg.n_vocab
d_model = model_cfg.d_model
seq_len = model_cfg.seq_len
n_layers = model_cfg.n_layers
n_heads = model_cfg.n_heads
n_kv_heads = model_cfg.n_kv_heads
d_hidden = model_cfg.d_hidden

# class RMSNorm(nn.Module):
    
#     def __init__(self, dim:int, eps:float = 1e-6) -> None:
#         super().__init__()
#         self.eps = eps
#         self.d_model = d_model
#         self.weights = nn.Parameter(torch.ones(dim))
    
#     def forward(self, x: Tensor) -> Tensor:
#         rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
#         return (x / rms) * self.weights

def precompute_rope_frequencies(
    head_dim: int = d_model // n_heads, 
    seq_len:int = seq_len, 
    base: float = 10000.0
) -> Tuple[Tensor, Tensor]:
    theta = 1 / (base ** (torch.arange(0, head_dim, 2).float() / d_model))
    positions = torch.arange(0, seq_len).float()
    angles = torch.outer(positions, theta)
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    seq_len = x.shape[2]

    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    return torch.stack([out1, out2], dim=-1).flatten(-2)

# class LiteGPT(nn.Module):

#     def __init__(self) -> None:
#         self.token_emb = nn.Embedding(n_vocab, d_model)
