"""Text generation with the LiteGPT model."""

import torch
from pathlib import Path
from omegaconf import OmegaConf
import tiktoken
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.model import LiteGPT

cfg = OmegaConf.load(
    Path(__file__).resolve().parent.parent / "configs/data/shakespeare.yaml"
)

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
            tokens = torch.cat([tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

    # Decode generated tokens
    generated_text = tokenizer.decode(generated_tokens)
    return generated_text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate text with LiteGPT")
    parser.add_argument(
        "--config",
        type=str,
        default="../../configs/train/LiteGPT-Small.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../../checkpoints/best",
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
        default="../../results/generated_text.txt",
        help="Output file for generated text",
    )

    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)
    device = config.device

    # Load model
    model = LiteGPT()
    model.to(device)

    # Load checkpoint (supports .safetensors, .pt, and checkpoint directories)
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        if ckpt_path.suffix == ".safetensors":
            from safetensors.torch import load_file as safetensors_load

            state_dict = safetensors_load(str(ckpt_path))
        elif ckpt_path.is_dir():
            from safetensors.torch import load_file as safetensors_load

            state_dict = safetensors_load(str(ckpt_path / "model.safetensors"))
        else:
            checkpoint = torch.load(ckpt_path, weights_only=True)
            state_dict = checkpoint["model_state_dict"]
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {args.checkpoint}")
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
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    full_text = args.prompt + generated
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
