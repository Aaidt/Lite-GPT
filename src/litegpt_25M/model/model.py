import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from typing import Tuple, cast
from omegaconf import OmegaConf


model_cfg = OmegaConf.load("./configs/model/LiteGPT-25M.yaml")
n_vocab = model_cfg.n_vocab
d_model = model_cfg.d_model
seq_len = model_cfg.seq_len
n_layers = model_cfg.n_layers
n_heads = model_cfg.n_heads
n_kv_heads = model_cfg.n_kv_heads
d_hidden = model_cfg.d_hidden
dropout = model_cfg.dropout

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
    theta = 1 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
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

def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1:
        return x
    b, n_kv, seq, hd = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv, n_rep, seq, hd)
        .reshape(b, n_kv * n_rep, seq, hd)
    )

class GQA(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model should be completely divisible by n_heads"
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        
        # (B, T, C) => (B, T, nh * hd)
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim)

        # (B, T, C) => (B, T, nkvh * hd)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)

        self.out_proj = nn.Linear(d_model, d_model)

        cos, sin = precompute_rope_frequencies(self.head_dim, seq_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x: Tensor) -> Tensor:
        q = self.q_proj(x)  
        k = self.k_proj(x)  
        v = self.v_proj(x)  

        b, seq, _ = x.shape

        q = q.view(b, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cast(Tensor, self.rope_cos), cast(Tensor, self.rope_sin))
        k = apply_rope(k, cast(Tensor, self.rope_cos), cast(Tensor, self.rope_sin))

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(b, seq, d_model)
        )
        out = self.out_proj(out)

        return out

class SWiGLU_ffn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_hidden)
        self.w_up = nn.Linear(d_model, d_hidden)
        self.w_down = nn.Linear(d_hidden, d_model)

    def forward(self, x: Tensor):
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return F.dropout(self.w_down(gate * up), p=dropout, training=self.training)

class TransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.attention = GQA()
        self.ffn = SWiGLU_ffn()

    def forward(self, x: Tensor):
        x = x + self.attention(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

class LiteGPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(n_vocab, d_model)

        self.layers = nn.ModuleList(
            [TransformerBlock() for _ in range(n_layers)]
        )

        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, n_vocab, bias=False)

        self.apply(self._init_weights)
        self.token_emb.weight = self.lm_head.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, x: Tensor, targets: Tensor | None = None
    ) -> Tuple[Tensor, Tensor | None]:
        x = self.token_emb(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss
