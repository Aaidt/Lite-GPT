"""Benchmark LiteGPT model performance."""
import torch
import time
import json
from pathlib import Path
from omegaconf import OmegaConf
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.model.model import LiteGPT
from src.data.dataloader import LiteGPTDataLoader
from safetensors.torch import load_model


def benchmark_throughput(
    model: torch.nn.Module,
    dataloader: LiteGPTDataLoader,
    device: str,
    num_batches: int = 100,
) -> dict:
    """Benchmark model throughput (tokens per second).
    
    Args:
        model: Model to benchmark
        dataloader: Data loader
        device: Device to use
        num_batches: Number of batches to benchmark
        
    Returns:
        Dictionary with throughput metrics
    """
    model.eval()
    
    # Warmup
    for _ in range(5):
        x, y = dataloader.get_batch()
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            _ = model(x, y)
    
    # Measure
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    
    total_tokens = 0
    with torch.no_grad():
        for i in range(num_batches):
            x, y = dataloader.get_batch()
            x, y = x.to(device), y.to(device)
            
            _, _ = model(x, y)
            total_tokens += x.numel()
    
    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.time() - start_time
    
    throughput = total_tokens / elapsed
    
    return {
        "throughput_tokens_per_sec": throughput,
        "total_tokens": total_tokens,
        "elapsed_time_sec": elapsed,
        "num_batches": num_batches,
    }


def benchmark_latency(
    model: torch.nn.Module,
    dataloader: LiteGPTDataLoader,
    device: str,
    num_samples: int = 100,
) -> dict:
    """Benchmark model latency (ms per forward pass).
    
    Args:
        model: Model to benchmark
        dataloader: Data loader
        device: Device to use
        num_samples: Number of forward passes to measure
        
    Returns:
        Dictionary with latency metrics
    """
    model.eval()
    
    latencies = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            x, y = dataloader.get_batch()
            x, y = x.to(device), y.to(device)
            
            torch.cuda.synchronize() if device == "cuda" else None
            start = time.time()
            
            _, _ = model(x, y)
            
            torch.cuda.synchronize() if device == "cuda" else None
            elapsed = (time.time() - start) * 1000  # Convert to ms
            latencies.append(elapsed)
    
    latencies = sorted(latencies)
    
    return {
        "latency_mean_ms": sum(latencies) / len(latencies),
        "latency_median_ms": latencies[len(latencies) // 2],
        "latency_p95_ms": latencies[int(len(latencies) * 0.95)],
        "latency_p99_ms": latencies[int(len(latencies) * 0.99)],
        "num_samples": num_samples,
    }


def benchmark_memory(
    model: torch.nn.Module,
    dataloader: LiteGPTDataLoader,
    device: str,
) -> dict:
    """Benchmark GPU memory usage during inference."""

    if device != "cuda":
        return {
            "allocated_gb": "N/A",
            "reserved_gb": "N/A",
            "peak_allocated_gb": "N/A",
        }

    model.eval()

    x, y = dataloader.get_batch()
    x, y = x.to(device), y.to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with torch.no_grad():
        model(x, y)

    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    peak_allocated = torch.cuda.max_memory_allocated() / (1024**3)

    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "peak_allocated_gb": peak_allocated,
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark LiteGPT model")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/train/LiteGPT-Small.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/best",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/benchmark.json",
        help="Output file for benchmark results",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    device = config.device
    
    # Load model
    print("Loading model...")
    model = LiteGPT()
    model.to(device)
    
    # load checkpoint
    checkpoint_dir = Path(args.checkpoint)
    if checkpoint_dir.exists():
        load_model(
            model,
            str(checkpoint_dir / "model.safetensors")
        )
    
        training_state = torch.load(
            checkpoint_dir / "training_state.pt",
            weights_only=True,
        )
    
        print(
            f"Loaded checkpoint from step "
            f"{training_state['step']}"
        )
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}")

    # Load data
    dataloader = LiteGPTDataLoader(split="val")
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    
    print("- Memory benchmarking...")
    memory_metrics = benchmark_memory(model, dataloader, device)
    
    print("- Latency benchmarking...")
    latency_metrics = benchmark_latency(model, dataloader, device, num_samples=100)
    
    print("- Throughput benchmarking...")
    throughput_metrics = benchmark_throughput(model, dataloader, device, num_batches=100)
     
    # Combine all metrics
    results = {
        "memory": memory_metrics,
        "latency": latency_metrics,
        "throughput": throughput_metrics,
        "config": OmegaConf.to_container(config, resolve=True),
    }
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Benchmark Results")
    print("=" * 80)
    print(f"Memory (Allocated): {memory_metrics.get('allocated_gb', 'N/A')} GB")
    print(f"Latency (Mean): {latency_metrics['latency_mean_ms']:.2f} ms")
    print(f"Latency (Median): {latency_metrics['latency_median_ms']:.2f} ms")
    print(f"Latency (P95): {latency_metrics['latency_p95_ms']:.2f} ms")
    print(f"Throughput: {throughput_metrics['throughput_tokens_per_sec']:.0f} tokens/sec")
    print("=" * 80)
    print(f"Results saved to {args.output}")
