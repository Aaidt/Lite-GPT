import torch
import time
from typing import cast, Any
from omegaconf import OmegaConf
from data.dataloader import LiteGPTDataLoader
from model.model import LiteGPT
from scheduler import scheduler
from utils.logger import WandBLogger
from utils.metrics import TrainingMetrics

train_cfg = OmegaConf.load("../../configs/train/LiteGPT-Small.yaml")
device = train_cfg.device

# Initialize wandb logger
config = cast(
    dict[str, Any],
    OmegaConf.to_container(train_cfg, resolve=True)
)
logger = WandBLogger(
    project="Lite-GPT",
    entity="aadit_123-aadit",
    config=config,
    name="LiteGPT-Small-Training",
)
logger.start()

# Initialize metrics tracker
metrics = TrainingMetrics()

torch.manual_seed(train_cfg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(train_cfg.seed)

train_loader = LiteGPTDataLoader(split="train")
val_loader = LiteGPTDataLoader(split="val")


@torch.no_grad()
def estimate_loss():
    """Estimate validation loss."""
    model.eval()
    losses = []

    for i in range(train_cfg.eval_iters):
        x, y = val_loader.get_batch()
        x, y = x.to(device), y.to(device)

        _, loss = model(x, y)
        losses.append(loss.item())
        
    model.train()
    val_loss = sum(losses) / len(losses)
    metrics.add_val_loss(val_loss)
    return val_loss


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

print("=" * 80)
print("Training Configuration")
print("=" * 80)
print(OmegaConf.to_yaml(train_cfg))
print("=" * 80)

model.train()
accumulated_loss = 0.0
grad_accum_steps = train_cfg.grad_accum_steps

for i in range(train_cfg.max_iters):
    t0 = time.time()

    x, y = train_loader.get_batch()
    x, y = x.to(device), y.to(device)

    # Forward pass with gradient accumulation
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = compiled_model(x, y)
    
    # Scale loss by accumulation steps
    scaled_loss = loss / grad_accum_steps
    scaled_loss.backward()
    accumulated_loss += loss.item()

    # Gradient accumulation step
    if (i + 1) % grad_accum_steps == 0:
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm=train_cfg.grad_clip
        )
        
        lr = scheduler(i)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        optimizer.step()
        optimizer.zero_grad()
        
        # Average accumulated loss
        avg_loss = accumulated_loss / grad_accum_steps
        accumulated_loss = 0.0
        
        t1 = time.time()
        tokens_per_sec = (train_loader.batch_size * train_loader.seq_len * grad_accum_steps) / (t1 - t0)
        
        # Track metrics
        metrics.add_train_step(i, avg_loss, lr, norm.item(), tokens_per_sec)
        
        # Validation and logging
        if (i % train_cfg.eval_interval) == 0:
            val_loss = estimate_loss()
            
            log_dict = {
                "train/loss": avg_loss,
                "train/learning_rate": lr,
                "train/grad_norm": norm.item(),
                "train/tokens_per_sec": tokens_per_sec,
                "val/loss": val_loss,
                "train/step": i,
            }
            
            # Log to wandb
            logger.log(log_dict, step=i)
            
            print(
                f"[Step {i:5d}] "
                f"train_loss: {avg_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"lr: {lr:.2e} | "
                f"grad_norm: {norm:.2f} | "
                f"tok/s: {tokens_per_sec:.0f}"
            )
        else:
            log_dict = {
                "train/loss": avg_loss,
                "train/learning_rate": lr,
                "train/grad_norm": norm.item(),
                "train/tokens_per_sec": tokens_per_sec,
                "train/step": i,
            }
            logger.log(log_dict, step=i)
            
            print(
                f"[Step {i:5d}] "
                f"train_loss: {avg_loss:.4f} | "
                f"lr: {lr:.2e} | "
                f"grad_norm: {norm:.2f} | "
                f"tok/s: {tokens_per_sec:.0f}"
            )

print("=" * 80)
print("Training Complete")
print("=" * 80)
logger.finish()