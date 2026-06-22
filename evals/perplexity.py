"""Compute perplexity for language model evaluation."""
import torch
from pathlib import Path
from omegaconf import OmegaConf
import sys
import math

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.model import LiteGPT
from data.dataloader import LiteGPTDataLoader


def compute_perplexity(
    model: torch.nn.Module,
    dataloader: LiteGPTDataLoader,
    device: str,
    num_batches: int = 100,
) -> float:
    """Compute perplexity on a dataset.
    
    Args:
        model: Language model to evaluate
        dataloader: Data loader for evaluation
        device: Device to use for computation
        num_batches: Number of batches to evaluate, None for all
        
    Returns:
        Perplexity (lower is better)
    """
    model.eval()
    total_loss = 0.0
    num_batches_processed = 0
    
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = dataloader.get_batch()
            x, y = x.to(device), y.to(device)
            
            _, loss = model(x, y)
            total_loss += loss.item()
            num_batches_processed += 1
    
    # Perplexity = exp(average loss)
    if num_batches_processed == 0:
        raise ValueError("No batches processed")
    avg_loss = total_loss / num_batches_processed
    perplexity = math.exp(avg_loss)
    
    return perplexity


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute model perplexity")
    parser.add_argument(
        "--config",
        type=str,
        default="../../configs/train/LiteGPT-Small.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../../checkpoints/checkpoint_best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Which split to evaluate",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=100,
        help="Number of batches to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../../results/perplexity.txt",
        help="Output file for results",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    device = config.device
    
    # Load model
    model = LiteGPT()
    model.to(device)
    
    # Load checkpoint if available
    if Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}, using untrained model")
    
    # Load data
    dataloader = LiteGPTDataLoader(split=args.split)
    
    # Compute perplexity
    print(f"Computing perplexity on {args.split} set...")
    perplexity = compute_perplexity(
        model,
        dataloader,
        device,
        num_batches=args.num_batches,
    )
    
    print(f"Perplexity: {perplexity:.2f}")
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(f"{args.split.capitalize()} Perplexity: {perplexity:.2f}\n")
    
    print(f"Results saved to {args.output}")
