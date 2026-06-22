import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple
from omegaconf import OmegaConf, DictConfig, ListConfig

model_cfg = OmegaConf.load("./configs/model/LiteGPT-50M.yaml")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: DictConfig | ListConfig = model_cfg) -> None:
        super().__init__()

        self.config = cfg
        self.dropout = cfg.dropout

        self.n_head = self.config.n_head
        self.d_model = self.config.d_model
        self.head_dim = self.d_model // self.n_head

        assert self.d_model % self.n_head == 0, (
            "d_model / n_head should be divisible and give an int"
        )

        self.c_attn = nn.Linear(self.d_model, 3 * self.d_model, bias=self.config.bias)
        self.c_proj = nn.Linear(self.d_model, self.d_model, bias=self.config.bias)

        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # (B, T, C) => (B, nh, T, hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)

        return y


class MLP(nn.Module):
    def __init__(self, cfg: DictConfig | ListConfig = model_cfg) -> None:
        super().__init__()

        self.config = cfg
        self.d_model = self.config.d_model

        self.c_fc = nn.Linear(self.d_model, 4 * self.d_model, bias=self.config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * self.d_model, self.d_model, bias=self.config.bias)

        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

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
        self.seq_len = self.config.seq_len
        self.n_layers = self.config.n_layers

        self.token_emb = nn.Embedding(self.config.n_vocab, self.config.d_model)
        self.pos_emb = nn.Embedding(self.config.seq_len, self.config.d_model)

        self.transformer = nn.ModuleList(
            [TransformerBlock(self.config) for _ in range(self.n_layers)]
        )

        self.ln_f = nn.LayerNorm(self.config.d_model)

        self.lm_head = nn.Linear(self.config.d_model, self.config.n_vocab, bias=False)

        self.lm_head.weight = self.token_emb.weight

    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> Tuple[Tensor, Tensor | None]:
        B, T = idx.size()
        assert T <= self.seq_len, (
            f"Cannot forward sequence of size {T}, max seq_len is: {self.seq_len}"
        )

        positions = torch.arange(0, T, dtype=torch.long, device=idx.device)
        token_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(positions)

        x = token_emb + pos_emb

        for block in self.transformer:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss: Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

        return logits, loss
