import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from omegaconf import OmegaConf, DictConfig, ListConfig
from dataclasses import dataclass

model_cfg = OmegaConf.load("../../configs/model/LiteGPT-Small.yaml")


@dataclass
class ModelConfig:
    d_model: int
    n_layers: int
    n_vocab: int
    seq_len: int


class CasualSelfAttention(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()


class MLP(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()


class TransformerBlock(nn.Module):

    def __init__(self, cfg: DictConfig | ListConfig = model_cfg) -> None:
        super().__init__()
        


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