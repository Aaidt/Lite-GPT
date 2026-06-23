"""Text generation with the LiteGPT model."""

import torch
from pathlib import Path
from omegaconf import OmegaConf
import tiktoken
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.litegpt_25M.model.model import LiteGPT
from safetensors.torch import load_model

cfg = OmegaConf.load("./configs/data/LiteGPT-25M.yaml")

encoder = tiktoken.get_encoding(cfg.tokenizer)
assert encoder.decode(encoder.encode("Hello world")) == "Hello world", (
    "Tokenizer round-trip failed"
)

assert encoder.n_vocab <= np.iinfo(np.uint16).max, "n_vocab is more than uint16"


def generate(
    model: torch.nn.Module,
    prompt: str,
    tokenizer=encoder,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: str = "cuda",
) -> str:
    """Generate text from a prompt.

    Args:
        model: Language model
        prompt: Initial text prompt
        tokenizer: Tokenizer for encoding/decoding
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Use top-k sampling if specified
        device: Device to use

    Returns:
        Generated text
    """
    model.eval()

    # Encode prompt
    tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
    tokens = tokens.unsqueeze(0)  # Add batch dimension

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get model output
            logits, _ = model(tokens, None)

            # Use last token logits
            next_logits = logits[0, -1, :] / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                indices_to_remove = (
                    next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                )
                next_logits[indices_to_remove] = float("-inf")

            # Sample next token
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Check for end of sequence
            if next_token.item() == tokenizer.eot_token:
                break

            generated_tokens.append(next_token.item())
            tokens = torch.cat([tokens, next_token.view(1, -1)], dim=1)

    # Decode generated tokens
    generated_text = tokenizer.decode(generated_tokens)
    return generated_text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate text with LiteGPT")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/train/LiteGPT-25M.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/best",
        help="Path to model checkpoint (dir, .safetensors, or .pt)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox",
        help="Text prompt to continue",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Use top-k sampling",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/generated_text.txt",
        help="Output file for generated text",
    )

    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)
    device = config.device

    # Load model
    model = LiteGPT()
    model.to(device)

    # load checkpoint
    checkpoint_dir = Path(args.checkpoint)

    if checkpoint_dir.is_dir():
        if (checkpoint_dir / "model.safetensors").exists():
            load_model(
                model,
                str(checkpoint_dir / "model.safetensors")
            )
            print(f"Loaded model weights from {checkpoint_dir / 'model.safetensors'}")
            training_state = torch.load(
                checkpoint_dir / "training_state.pt",
                weights_only=True,
            )
            print(
                f"Loaded checkpoint from step "
                f"{training_state['step']}"
            )

    elif checkpoint_dir.suffix == ".safetensors":
        load_model(
            model,
            str(checkpoint_dir)
        )
        print(f"Loaded model weights from {checkpoint_dir}")

    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}")

    # Use the global tokenizer encoder
    tokenizer = encoder

    # Generate text
    print(f"Generating text from prompt: '{args.prompt}'")
    generated = generate(
        model,
        args.prompt,
        tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=max(args.temperature, 1e-5),  # Avoid zero temperature
        top_k=args.top_k,
        device=device,
    )

    full_text = args.prompt + generated

    metadata_file = checkpoint_dir / "metadata.json"

    if metadata_file.exists():
        import json
        with open(metadata_file) as f:
            metadata = json.load(f)

        print(
            f"Step: {metadata['step']} | "
            f"Val Loss: {metadata['metrics']['val_loss']:.4f}"
        )
        
    print("\n" + "=" * 80)
    print("Generated Text:")
    print("=" * 80)
    print(full_text)
    print("=" * 80)

    # Save to file
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(full_text)

    print(f"Generated text saved to {args.output}")
