import math
from omegaconf import OmegaConf

train_cfg = OmegaConf.load("./configs/train/LiteGPT-50M.yaml")

max_lr = train_cfg.max_lr
min_lr = max_lr * 0.01

max_steps = train_cfg.max_iters // train_cfg.grad_accum_steps
warmup_steps = train_cfg.warmup_iters // train_cfg.grad_accum_steps

def scheduler(it: int) -> float:
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1 
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)