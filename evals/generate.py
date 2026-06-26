"""Text generation with the LiteGPT model."""

import json
import torch
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
from tokenizers import Tokenizer
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.litegpt_25M.model.model import LiteGPT
from safetensors.torch import load_model

cfg = OmegaConf.load("./configs/data/LiteGPT-25M.yaml")

tokenizer = Tokenizer.from_file(str(Path(cfg.tokenizer_path)))
assert tokenizer.decode(tokenizer.encode("Hello world").ids) == "Hello world", (
    "Tokenizer round-trip failed"
)

assert tokenizer.get_vocab_size() <= np.iinfo(np.uint16).max, "n_vocab is more than uint16"
EOT_ID = tokenizer.token_to_id("<|endoftext|>")

CATEGORY_SETTINGS = {
    "General": {"temperature": 0.8, "top_k": 50, "max_tokens": 100},
    "Coding": {"temperature": 0.2, "top_k": 20, "max_tokens": 150},
    "Stories": {"temperature": 0.9, "top_k": 100, "max_tokens": 200},
    "Factual": {"temperature": 0.4, "top_k": 40, "max_tokens": 100},
    "Deterministic": {"temperature": 0.0, "top_k": None, "max_tokens": 50},
}

LONG_FORM_PROMPT = (
    "The Great Wall of China is a series of fortifications that were built across "
    "the historical northern borders of ancient Chinese states and Imperial China as "
    "protection against various nomadic groups from the Eurasian Steppe. Several walls "
    "were built from the 7th century BC, with selective stretches later joined together "
    "by Qin Shi Huang, the first emperor of China. Little of the Qin wall remains. "
    "Later on, many successive dynasties built and maintained multiple stretches of "
    "border walls. The best-known sections of the wall were built by the Ming dynasty."
)

EVAL_SUITE = [
    ("Language modeling", "General", [
        "The capital of France is",
        "Once upon a time, there lived",
        "The theory of evolution states that",
        "Machine learning is",
    ]),
    ("Reasoning", "Factual", [
        "If John has 5 apples and gives 2 to Alice, then John has",
        "The next number in the sequence is:\n2, 4, 8, 16,",
        "Earth is to Solar System as electron is to",
    ]),
    ("World knowledge", "Factual", [
        "Python is a programming language that",
        "The Internet works because",
        "The largest planet in our solar system is",
    ]),
    ("Coding", "Coding", [
        "def factorial(n):",
        "class LinkedList:",
        "for i in range(10):",
        "#include <iostream>\n\nint main() {",
        "function quickSort(arr) {",
    ]),
    ("HTML", "Coding", [
        "<!DOCTYPE html>\n<html>",
    ]),
    ("Markdown", "General", [
        "# Neural Networks",
    ]),
    ("Story completion", "Stories", [
        "The dragon opened its eyes and",
        "The detective walked into the abandoned warehouse and",
    ]),
    ("Dialogue", "Stories", [
        "Alice: Hello!\nBob:",
    ]),
    ("TinyStories style", "Stories", [
        "Tom had a little red ball.",
        "Lily loved going to the park because",
    ]),
    ("Long-form continuation", "General", [
        LONG_FORM_PROMPT,
    ]),
]


def run_eval_suite(model, tokenizer, device="cuda"):
    """Run the full evaluation suite through all preset prompts."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("./results/evals")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"eval_{timestamp}.md"

    lines = []
    lines.append("# LiteGPT Evaluation Suite")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    for category, settings_key, prompts in EVAL_SUITE:
        settings = CATEGORY_SETTINGS[settings_key]
        temp = settings["temperature"]
        topk = settings["top_k"]
        max_tok = settings["max_tokens"]

        lines.append(f"---\n")
        lines.append(f"## {category}")
        lines.append(f"- **Settings:** temperature={temp}, top_k={topk}, max_tokens={max_tok}")
        lines.append("")

        for prompt in prompts:
            effective_temp = 1e-5 if temp < 1e-5 else temp
            generated = generate(
                model, prompt, tokenizer,
                max_new_tokens=max_tok,
                temperature=effective_temp,
                top_k=topk,
                device=device,
            )

            full_text = prompt + generated

            lines.append(f"### Prompt: {prompt.replace(chr(10), ' ')}")
            lines.append(f"- **Output:** {full_text.replace(chr(10), ' ')}")
            lines.append("")

    output = "\n".join(lines)
    with open(output_file, "w") as f:
        f.write(output)

    return output_file


def generate(
    model: torch.nn.Module,
    prompt: str,
    tokenizer=tokenizer,
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
    tokens = torch.tensor(tokenizer.encode(prompt).ids, dtype=torch.long, device=device)
    tokens = tokens.unsqueeze(0)  # Add batch dimension

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get model output
            logits, _ = model(tokens, None)

            # Use last token logits
            next_logits = logits[0, -1, :]

            if temperature < 1e-5:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature

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
            if next_token.item() == EOT_ID:
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
        "--eval",
        action="store_true",
        help="Run full evaluation suite through all preset prompts",
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
        default=0,
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
    if checkpoint_dir.exists():
        if checkpoint_dir.is_dir():
            model_path = checkpoint_dir / "model.safetensors"
            if model_path.exists():
                load_model(model, str(model_path))
                print(f"  📂  Loaded model weights from {model_path}")
            else:
                print(f"  ⚠️  Warning: model file not found in checkpoint directory {checkpoint_dir}")

            training_state_path = checkpoint_dir / "training_state.pt"
            if training_state_path.exists():
                training_state = torch.load(
                    training_state_path,
                    weights_only=True,
                )
                print(f"  📍  Checkpoint step: {training_state['step']}")
        elif checkpoint_dir.suffix == ".safetensors":
            load_model(model, str(checkpoint_dir))
            print(f"  📂  Loaded model weights from {checkpoint_dir}")
        else:
            print(f"  ⚠️  Warning: Unsupported checkpoint file type: {checkpoint_dir}")
    else:
        print(f"  ⚠️  Warning: Checkpoint not found at {args.checkpoint}")

    if args.eval:
        output_file = run_eval_suite(model, tokenizer, device=device)
        print(f"\n  ✅  Evaluation suite saved to {output_file}")
        sys.exit(0)

    # Generate text
    print(f"\n  ✏️  Generating from prompt: \"{args.prompt}\"")
    print(f"  ⚙️   max_tokens={args.max_tokens}, temperature={args.temperature}, top_k={args.top_k}")
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
        with open(metadata_file) as f:
            metadata = json.load(f)

        print(f"  📊  Step: {metadata['step']} | Val Loss: {metadata['metrics']['val_loss']:.4f}")
        
    print("\n" + "╔" + "═" * 100 + "╗")
    print("║" + "          ✍️  Generated Text                  ".center(100) + "║")
    print("╠" + "═" * 100 + "╣")
    print("║ " + full_text.replace("\n", "\n║ ") + " " * max(0, 76 - len(full_text.split('\n')[-1])) + "║")
    print("╚" + "═" * 100 + "╝")

    # Save to file
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(full_text)

    print(f"  💾  Generated text saved to {args.output}")
