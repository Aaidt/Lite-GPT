# Model Architecture

## Overview
- Model type: Decoder-only Transformer
- Parameters: ~16M
- Context length: 128
- Vocabulary size: 50,257
- Attention: Causal Self-Attention
- Positional Encoding: Learned Position Embeddings

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
        ├──────────────┐
        ▼              │
┌─────────────────────┐│
│ Position Embeddings ││
│ [seq_len, d_model]  ││
└─────────────────────┘│
        │              │
        └──────┬───────┘
               ▼
          x = tok + pos
               │
               ▼
╔══════════════════════════════╗
║ Transformer Block × 4        ║
║                              ║
║  LayerNorm                   ║
║      │                       ║
║      ▼                       ║
║  Multi-Head Attention        ║
║      │                       ║
║      ▼                       ║
║  Residual Add                ║
║                              ║
║  LayerNorm                   ║
║      │                       ║
║      ▼                       ║
║     FFN                      ║
║      │                       ║
║      ▼                       ║
║  Residual Add                ║
╚══════════════════════════════╝
            │
            ▼
┌─────────────────────┐
│   Final LayerNorm   │
└─────────────────────┘
            │
            ▼
┌─────────────────────┐
│      LM Head        │
│  Weight Tied with   │
│ Token Embeddings    │
└─────────────────────┘
            │
            ▼
      Logits [B,T,V]
```

## Configuration

| Parameter | Value |
|------------|---------|
| n_layers | 4 |
| d_model | 256 |
| n_heads | 4 |
| head_dim | 64 |
| ffn_dim | 1024 |
| context_length | 128 |
| vocab_size | 50257 |

## Transformer Block

### Attention
- Multi-Head Self Attention
- Causal Masking

### Feed Forward Network

FFN(x) = W2(GELU(W1(x)))

Expansion ratio: 4×

### Residual Connections

x = x + Attention(x)

x = x + FFN(x)

### Normalization
- LayerNorm

## Parameter Count

| Component | Params |
|------------|---------|
| Token Embeddings | (n_vocab x d_model) 50257 x 256 = 12,865,792 |
| Position Embeddings | (seq_len x d_model) 128 x 256 = 32,768 |
| Attention | ([QKV + O] x n_layers) [256 x (3 x 256) + 256 x 256] x 4 = 1,048,576 |
| FFN | [(d_model x (4 x d_model) + (4 x d_model) + d_model)] x n_layers = [256 x 1024 + 1024 x 256] x 4 = 2,097,152 |
| Norm | [(2 x d_model) x n_layers] (2 x 256 x 4) = 2048 |
| Final Norm | 256 |
| LM Head | weight tying with token embeddings |
| Total | ~16M |

## Design Decisions

This model is intentionally kept as close to GPT-2 as possible to build a strong understanding of decoder-only transformers before introducing modern architectural improvements.

### GPT-2 Baseline
The model uses:
- Learned token embeddings
- Learned positional embeddings
- Multi-Head Self Attention (MHSA)
- GELU activations
- LayerNorm
- Causal masking

### Simplicity Over Performance
Features such as RoPE, GQA, FlashAttention, SwiGLU, RMSNorm, and Mixture-of-Experts are intentionally omitted. While these improve efficiency or performance, they add implementation complexity and make it harder to study the core transformer architecture.

### Small Scale Training
The model is designed to train on a single NVIDIA T4 GPU using Google Colab. Model size, context length, and batch size are chosen to fit within limited compute resources.

### Educational Focus
The goal of LiteGPT is not to achieve state-of-the-art performance, but to provide a clean and understandable implementation of a GPT-style language model that can be trained from scratch and extended with modern techniques in future experiments.