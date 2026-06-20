import torch
import time
from omegaconf import OmegaConf
from data.dataloader import LiteGPTDataLoader
from model.model import LiteGPT
from scheduler import scheduler

train_cfg = OmegaConf.load("../../configs/train/LiteGPT-Small.yaml")
device = train_cfg.device

torch.manual_seed(train_cfg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(train_cfg.seed)

train_loader = LiteGPTDataLoader(split="train")

torch.set_float32_matmul_precision("medium")

model = LiteGPT()
model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_cfg.max_lr,
    betas=(train_cfg.beta1, train_cfg.beta2),
    eps=train_cfg.eps
)

compiled_model = torch.compile(model)

model.train()
for i in range(train_cfg.max_iters):
    t0 = time.time()

    x, y = train_loader.get_batch()
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()

    logits, loss = compiled_model(x, y)

    loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(compiled_model.parameters(), max_norm=1.0)
    
    optimizer.step()

    t1 = time.time()

    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.batch_size * train_loader.seq_len) / (t1 - t0)

    print(f"Step: {i} | Loss: {loss.item()} | norm: {norm:.4f} | dt: {dt:.2f}ms | tokens_per_sec: {tokens_per_sec:.2f}")