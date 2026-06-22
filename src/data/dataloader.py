import torch
from typing import Literal, Tuple
from omegaconf import OmegaConf
import numpy as np

model_cfg = OmegaConf.load("./configs/model/LiteGPT-50M.yaml")
train_cfg = OmegaConf.load("./configs/train/LiteGPT-50M.yaml")
data_cfg = OmegaConf.load("./configs/data/shakespeare.yaml")

class LiteGPTDataLoader:

    def __init__(
        self, 
        split: Literal["train", "val"], 
        batch_size: int = train_cfg.batch_size,
        seq_len: int = model_cfg.seq_len
    ) -> None:
        
        if split == "train":
            path = data_cfg.train_bin
        else:
            path = data_cfg.val_bin
            
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.data = np.memmap(
            path,
            dtype=np.uint16,
            mode="r"
        )

    
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(
            len(self.data) - self.seq_len,
            (self.batch_size,)
        )

        x = torch.stack([
            torch.from_numpy(
                self.data[i:i+self.seq_len]
                .astype(np.int64)
            )
            for i in ix.tolist()
        ])

        y = torch.stack([
            torch.from_numpy(
                self.data[i+1 : i+self.seq_len+1]
                .astype(np.int64)
            )
            for i in ix.tolist()
        ])

        assert x.shape == (self.batch_size, self.seq_len), "x has the wrong tensor shape, it should (batch_size, seq_len)"
        assert y.shape == (self.batch_size, self.seq_len), "y has the wrong tensor shape, it should (batch_size, seq_len)"

        return x, y