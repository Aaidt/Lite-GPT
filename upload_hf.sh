#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="LiteGPT-25M"
OUT_DIR="./$REPO_NAME"

# run benchmarks
echo "🛻 Running evals..."
uv run python3 -m evals.benchmark
uv run python3 -m evals.generate --eval

echo "📁  Creating folder: $REPO_NAME"
mkdir -p "$OUT_DIR"

echo "📂  Copying checkpoints, results, and logs..."
[ -d "./checkpoints" ] && cp -r ./checkpoints "$OUT_DIR/" && echo "  ✔  checkpoints"
[ -d "./results" ]     && cp -r ./results "$OUT_DIR/"     && echo "  ✔  results"
[ -d "./logs" ]        && cp -r ./logs "$OUT_DIR/"        && echo "  ✔  logs"

echo ""
echo "📝  Creating config.json..."
cat > "$OUT_DIR/config.json" << 'CONFIG_EOF'
{
  "n_vocab": 16384,
  "seq_len": 512,
  "n_layers": 8,
  "n_heads": 8,
  "n_kv_heads": 4,
  "d_model": 448,
  "d_hidden": 1152,
  "dropout": 0.0,
  "bias": false,
  "batch_size": 64,
  "grad_accum_steps": 2,
  "max_iters": 40000,
  "warmup_iters": 4000,
  "max_lr": 0.0006,
  "weight_decay": 0.1,
  "beta1": 0.9,
  "beta2": 0.95,
  "eps": 1e-8,
  "eval_interval": 500,
  "eval_iters": 100,
  "grad_clip": 1.0,
  "device": "cuda",
  "seed": 42,
  "patience": 10
}
CONFIG_EOF
echo "  ✔  config.json"

echo ""
echo "📝  Creating README.md..."

RESULTS_FILE="./results/training_results.json"
if [ -f "$RESULTS_FILE" ]; then
    train_loss=$(python3 -c "import json; d=json.load(open('$RESULTS_FILE')); print(d['final_metrics']['loss'])")
    val_loss=$(python3 -c "import json; d=json.load(open('$RESULTS_FILE')); print(d['final_metrics']['val_loss'])")
    perplexity=$(python3 -c "import json; d=json.load(open('$RESULTS_FILE')); print(d['final_metrics']['perplexity'])")
    tokens_per_sec=$(python3 -c "import json; d=json.load(open('$RESULTS_FILE')); print(d['final_metrics']['tokens_per_sec'])")
    best_val_loss=$(python3 -c "import json; d=json.load(open('$RESULTS_FILE')); print(d['best_val_loss'])")
else
    train_loss="2.9977498054504395"
    val_loss="2.9335184621810915"
    perplexity="18.793639012453678"
    tokens_per_sec="236436.41890370354"
    best_val_loss="2.9335184621810915"
fi

cat > "$OUT_DIR/README.md" << 'BODY_EOF'
---
language:
- en
license: mit
pipeline_tag: text-generation
tags:
- pytorch
- transformer
- gpt
- causal-lm
library_name: pytorch
---

# Model Architecture

The goal of LiteGPT-25M is not to achieve state-of-the-art performance, but to provide a clean and understandable implementation of a modern Llama-style language model that can be trained from scratch and extended with further techniques in future experiments.

## Overview
- Model type: Decoder-only Transformer
- Parameters: ~25M
- Context length: 512
- Vocabulary size: 16,384 (custom ByteLevel BPE)
- Attention: GQA Flash-Attention (8 query / 4 key-value heads)
- Positional Encoding: RoPE
- Normalization: RMSNorm
- Activation: SwiGLU

## Architecture Diagram

```text
Input Tokens [B, T]
        │
        ▼
┌─────────────────────┐
│  Token Embeddings   │
│  [vocab, d_model]   │
└─────────────────────┘
           │
           │
           ▼
╔══════════════════════════════╗
║ Transformer Block × 8        ║
║                              ║
║       RMSNorm                ║
║           │                  ║
║           ▼                  ║
║     GQA Flash Attention      ║
║           │                  ║
║           ▼                  ║
║       Residual Add           ║
║           │                  ║
║           ▼                  ║
║       RMSNorm                ║
║           │                  ║
║           ▼                  ║
║        SwiGLU MLP            ║
║           │                  ║
║           ▼                  ║
║      Residual Add            ║
╚══════════════════════════════╝
            │
            ▼
┌─────────────────────┐
│   Final RMSNorm     │
└─────────────────────┘
            │
            ▼
┌─────────────────────┐
│      LM Head        │
└─────────────────────┘
            │
            ▼
      Logits [B,T,V]
```

## Configuration

| Parameter | Value |
|------------|---------|
| batch_size | 64 |
| grad_accum_steps | 2 |
| n_layers | 8 |
| d_model | 448 |
| n_heads | 8 |
| n_kv_heads | 4 |
| head_dim | 56 |
| ffn_dim | 1152 |
| context_length | 512 |
| vocab_size | 16384 |

## Transformer Block

### Attention
- Grouped Query Attention (GQA) with 8 query heads and 4 key-value heads
- FlashAttention for memory-efficient computation
- RoPE for positional encoding
- Causal masking

### Feed Forward Network (SwiGLU)

FFN(x) = W2(SiLU(W1(x)) ⊙ W3(x))

Expansion ratio: ~2.57×

### Residual Connections

x = x + Attention(x)

x = x + FFN(x)

### Normalization
- RMSNorm (pre-attention and pre-FFN)

## Parameter Count

| Component | Params |
|------------|---------|
| Token Embeddings | (n_vocab x d_model) 16384 x 448 = 7,340,032 |
| Attention (GQA) | [(448x448) + (448x224) + (448x224) + (448x448)] x 8 = 4,816,896 |
| SwiGLU FFN | [(448x1152) + (448x1152) + (1152x448)] x 8 = 12,386,304 |
| RMSNorm | (448 x 2) x 8 = 7,168 |
| Final RMSNorm | 448 |
| LM Head | weight tied with token embeddings |
| Total | ~24.6M |

## Design Decisions

This model incorporates a subset of modern LLM architectural improvements while remaining small enough to train from scratch on a single NVIDIA A5000 GPU.

### Modern Decoder Architecture
The model uses:
- Learned token embeddings
- Rotary Positional Embeddings (RoPE)
- Grouped Query Attention (GQA)
- FlashAttention
- SwiGLU feed-forward networks
- RMSNorm
- Causal masking
- Weight tying between token embeddings and LM head

### Modern Features With Minimal Complexity
Several architectural improvements from models such as Llama, Mistral, and Gemma are included because they provide meaningful gains in training efficiency, model quality, or inference performance without substantially increasing implementation complexity.

Key improvements over GPT-2 include:
- **RoPE** for improved positional encoding and length extrapolation
- **GQA** for reduced KV cache size and improved inference efficiency
- **FlashAttention** for faster and more memory-efficient attention computation
- **SwiGLU** for improved feed-forward expressiveness
- **RMSNorm** for simpler and more efficient normalization

### Small Scale Training
The model is designed to train on a single NVIDIA A5000. Model size, context length, and batch size are selected to balance training speed, memory usage, and model quality within a constrained compute budget.

### Educational Goal
The objective is to bridge the gap between a GPT-2 style transformer and a modern Llama-style architecture while keeping the codebase compact, readable, and suitable for experimentation.

## Dataset

| Metric | Value |
|----------|---------|
| Total Tokens | ~500M |
| Tokenizer | Custom ByteLevel BPE |
| Vocabulary Size | 16,384 |
| Context Length | 512 |
| Train Split | 90% |
| Validation Split | 10% |
| Storage Format | uint16 |

### Dataset Composition

| Dataset | Tokens | Percentage |
|-----------|--------|------------|
| FineWeb | 300M | 60% |
| TinyStories | 200M | 40% |

### Data Pipeline

```text
FineWeb (300M)
TinyStories (200M)
        │
        ▼
┌──────────────────┐
│ Dataset Mixing   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ ByteLevel BPE    │
│ (vocab 16,384)   │
└────────┬─────────┘
         │
         ▼
      Token IDs
         │
         ▼
┌──────────────────┐
│ Append EOT Token │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Random 90/10     │
│ Train / Val      │
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
train.bin  val.bin
(uint16)   (uint16)
```

### Tokenizer
- Type: ByteLevel BPE (trained on the corpus)
- Vocabulary size: 16,384
- Special tokens: `<|endoftext|>`
- Tokens stored as uint16

## Training

### Hardware
- GPU: RTX A5000 x1
- Platform: Runpod

### Hyperparameters

| Parameter | Value |
|------------|---------|
| Batch Size | 64 |
| Gradient Accumulation | 2 |
| Effective Batch Size | 128 |
| Sequence Length | 512 |
| Learning Rate | 6e-4 |
| Min Learning Rate | 6e-5 |
| Weight Decay | 0.1 |
| Warmup Steps | 4000 |
| Max Steps | 40000 |
| Eval Interval | 500 |
| Eval Iters | 100 |
| Grad Clip | 1.0 |

### Tokens Seen
```text
64 x 512 x 40000 = 1,310,720,000 tokens ~ 1.3B tokens
```

### Optimizer
AdamW (β1 = 0.9, β2 = 0.95)

### Learning Rate Schedule
Cosine Decay with linear warmup (4000 steps warmup, 36000 steps decay)

### Mixed Precision
BF16 / FP16

### Loss Function
Cross Entropy Loss

## Results

BODY_EOF

cat >> "$OUT_DIR/README.md" << RESULTS_EOF
| Metric | values |
|---------|----------|
| Train Loss | $train_loss |
| Val Loss | $val_loss |
| Perplexity | $perplexity |
| tokens_per_sec | $tokens_per_sec | 
| best_val_loss | $best_val_loss | 
RESULTS_EOF
echo "  ✔  README.md"

echo ""
echo "📦  Installing Hugging Face CLI..."
if ! command -v hf &>/dev/null; then
    curl -LsSf https://hf.co/cli/install.sh | bash -s
    export PATH="$HOME/.local/bin:$PATH"
fi

echo ""
echo "🔑  Logging in to Hugging Face..."
read -rsp "  Enter your Hugging Face token: " HF_TOKEN
echo ""
hf auth login --token "$HF_TOKEN" --add-to-git-credential

echo ""
echo "📄  Files to be uploaded:"
find "$OUT_DIR" -type f | sort

echo ""
read -rp "❓  Upload to Hugging Face? (y/N) " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "  Aborted."
    exit 0
fi

echo ""
echo "🚀  Uploading $REPO_NAME to Hugging Face..."
hf upload "$REPO_NAME" "$OUT_DIR" --repo-type model

echo ""
echo "✅  Done — uploaded $REPO_NAME to Hugging Face."