import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
from omegaconf import OmegaConf, DictConfig, ListConfig

model_cfg = OmegaConf.load("../../configs/model/LiteGPT-Small.yaml")


class CausalSelfAttention(nn.Module):
    
    def __init__(self, cfg: DictConfig | ListConfig = model_cfg) -> None:
        super().__init__()

        self.config = cfg
        self.dropout = cfg.dropout

        self.n_head = self.config.n_head
        self.d_model = self.config.d_model
        self.head_dim = self.d_model // self.n_head

        assert self.d_model % self.n_head == 0, "d_model / n_head should be divisible and give an int"

        self.c_attn = nn.Linear(self.d_model, 3 * self.d_model)
        self.c_proj = nn.Linear(self.d_model, self.d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # (B, T, C) => (B, nh, T, hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, 
            k, 
            v, 
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y



class MLP(nn.Module):
    
    def __init__(self, cfg: DictConfig | ListConfig = model_cfg) -> None:
        super().__init__()

        self.config = cfg
        self.d_model = self.config.d_model
        
        self.c_fc = nn.Linear(self.d_model, 4 * self.d_model)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * self.d_model, self.d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x


class TransformerBlock(nn.Module):

    def __init__(self, cfg: DictConfig | ListConfig = model_cfg) -> None:
        super().__init__()

        self.config = cfg
        self.d_model = self.config.d_model

        self.ln_1 = nn.LayerNorm(self.d_model)
        self.attn = CausalSelfAttention(self.config)
        self.ln_2 = nn.LayerNorm(self.d_model)
        self.mlp = MLP(self.config)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


class LiteGPT(nn.Module):

    def __init__(self, cfg: DictConfig | ListConfig = model_cfg) -> None:
        super().__init__()
        self.config = cfg

        self.token_emb = nn.Embedding(self.config.n_vocab, self.config.d_model)
        self.pos_emb = nn.Embedding(self.config.seq_len, self.config.d_model)

        self.transformer = nn.ModuleList(
            [TransformerBlock() for _ in range(self.config.n_layers)]
        )

        self.layer_norm = nn.LayerNorm(self.config.d_model)

        self.lm_head = nn.Linear(self.config.d_model, self.config.n_vocab, bias=False)

        # weight tying
        self.token_emb.weight = self.lm_head.weight