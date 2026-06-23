import torch
import math
import time
from pathlib import Path
from typing import cast, Any
from omegaconf import OmegaConf

from src.litegpt_16M.data.dataloader import LiteGPTDataLoader
from src.litegpt_16M.model.model import LiteGPT
from src.litegpt_16M.utils.logger import WandBLogger
from src.litegpt_16M.utils.metrics import TrainingMetrics
from .scheduler import scheduler
from .checkpointing import CheckpointManager

train_cfg = OmegaConf.load("./configs/train/LiteGPT-50M.yaml")
device = train_cfg.device
max_steps = train_cfg.max_iters // train_cfg.grad_accum_steps
warmup_steps = train_cfg.warmup_iters // train_cfg.grad_accum_steps

# Create directories
logs_dir = Path("./content/drive/MyDrive/LiteGPT/logs")
results_dir = Path("./content/drive/MyDrive/LiteGPT/results")
checkpoints_dir = Path("./content/drive/MyDrive/LiteGPT/checkpoints")
logs_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

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

# Initialize checkpoint manager
checkpoint_manager = CheckpointManager(checkpoint_dir=str(checkpoints_dir))

# Initialize metrics tracker
metrics = TrainingMetrics()

# Store best validation loss
best_val_loss = float("inf")

torch.manual_seed(train_cfg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(train_cfg.seed)

train_loader = LiteGPTDataLoader(split="train")
val_loader = LiteGPTDataLoader(split="val")

model = LiteGPT()
model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_cfg.max_lr,
    betas=(train_cfg.beta1, train_cfg.beta2),
    eps=train_cfg.eps
)

compiled_model = torch.compile(model)

@torch.no_grad()
def estimate_loss():
    """Estimate validation loss."""
    model.eval()
    losses = []

    for _ in range(train_cfg.eval_iters):
        x, y = val_loader.get_batch()
        x, y = x.to(device), y.to(device)

        _, loss = compiled_model(x, y)
        losses.append(loss.item())
        
    model.train()
    val_loss = sum(losses) / len(losses)
    perplexity = math.exp(min(val_loss, 20)) # Cap loss to prevent overflow in exp
    metrics.add_val_loss(val_loss)
    metrics.add_perplexity(perplexity)
    return val_loss, perplexity


print("=" * 80)
print("Training Configuration")
print("=" * 80)
print(OmegaConf.to_yaml(train_cfg))
print("=" * 80)
print("=" * 80)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)
print(f"Total params:     {total_params:,}")
print(f"Trainable params: {trainable_params:,}")
print(f"Frozen params:    {total_params - trainable_params:,}")
print("=" * 80)

for name, module in model.named_modules():
    total = sum(p.numel()for p in module.parameters(recurse=False))

    trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)

    if total > 0:
        print(
            f"{name:30} "
            f"total={total:>10,} "
            f"trainable={trainable:>10,}"
        )
print("=" * 80)

model.train()
accumulated_loss = 0.0
grad_accum_steps = train_cfg.grad_accum_steps

step_start: float | None = None

assert train_cfg.max_iters % grad_accum_steps == 0, "grad_accum_steps should completely divide max_iters otherwise grads at the end will be ignored"

for i in range(train_cfg.max_iters):

    if i % grad_accum_steps == 0:
        if device == "cuda":
            torch.cuda.synchronize()
        step_start = time.time()

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
        
        optimizer_step = (i + 1) // grad_accum_steps
        lr = scheduler(optimizer_step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        optimizer.step()
        optimizer.zero_grad()
        
        # Average accumulated loss
        avg_loss = accumulated_loss / grad_accum_steps
        accumulated_loss = 0.0
        
        if device == "cuda":
            torch.cuda.synchronize()

        assert step_start is not None, "starting time must not be None"
        step_time = time.time() - step_start
        
        tokens_per_sec = (train_loader.batch_size * train_loader.seq_len * grad_accum_steps) / step_time
        
        # Track metrics
        metrics.add_train_step(optimizer_step, avg_loss, lr, norm.item(), tokens_per_sec)
        
        # Validation and logging
        if (optimizer_step % train_cfg.eval_interval) == 0:
            val_loss, perplexity = estimate_loss()
            
            log_dict = {
                "train/loss": avg_loss,
                "train/learning_rate": lr,
                "train/grad_norm": norm.item(),
                "train/tokens_per_sec": tokens_per_sec,
                "val/loss": val_loss,
                "val/perplexity": perplexity,
                "train/step": optimizer_step,
            }
            
            # Log to wandb
            logger.log(log_dict, step=optimizer_step)
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            checkpoint_manager.save(
                step=optimizer_step,
                model=model,
                optimizer=optimizer,
                metrics={
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "train_loss": avg_loss,
                    "val_loss": val_loss,
                    "val_perplexity": perplexity,
                    "learning_rate": lr,
                    "grad_norm": norm.item(),
                    "tokens_per_sec": tokens_per_sec,
                },
                is_best=is_best,
            )
            
            # Cleanup old checkpoints
            checkpoint_manager.delete_old_checkpoints(keep_last=3)
            
            print(
                f"[Step {i:5d}] "
                f"train_loss: {avg_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_perplexity: {perplexity:.2f} | "
                f"lr: {lr:.2e} | "
                f"grad_norm: {norm:.2f} | "
                f"tok/s: {tokens_per_sec:.0f}"
                f"decay_ratio={(optimizer_step - warmup_steps) / (max_steps - warmup_steps):.3f}"
                + (" [BEST]" if is_best else "")
            )
        else:
            log_dict = {
                "train/loss": avg_loss,
                "train/learning_rate": lr,
                "train/grad_norm": norm.item(),
                "train/tokens_per_sec": tokens_per_sec,
                "train/step": optimizer_step,
            }
            logger.log(log_dict, step=optimizer_step)
            
            print(
                f"[Step {i:5d}] "
                f"train_loss: {avg_loss:.4f} | "
                f"lr: {lr:.2e} | "
                f"grad_norm: {norm:.2f} | "
                f"tok/s: {tokens_per_sec:.0f}",
                f"train/step: {optimizer_step}"
            )

print("=" * 80)
print("Training Complete")
print("=" * 80)
logger.finish()

# Save final results
import json
results_file = results_dir / "training_results.json"
with open(results_file, "w") as f:
    json.dump({
        "final_metrics": metrics.get_latest_metrics(),
        "best_val_loss": best_val_loss,
        "total_optimizer_steps": train_cfg.max_iters // grad_accum_steps,
        "config": config,
    }, f, indent=2)

# Save metrics to logs
logs_file = logs_dir / "training_metrics.json"
with open(logs_file, "w") as f:
    json.dump({
        "losses": metrics.losses,
        "learning_rates": metrics.learning_rates,
        "grad_norms": metrics.grad_norms,
        "tokens_per_sec": metrics.tokens_per_sec,
        "val_losses": metrics.val_losses,
        "steps": metrics.steps,
    }, f, indent=2)

print(f"Results saved to: {results_file}")
print(f"Metrics saved to: {logs_file}")