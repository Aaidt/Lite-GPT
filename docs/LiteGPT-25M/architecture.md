# Model Architecture

The goal of LiteGPT-25M is not to achieve state-of-the-art performance, but to provide a clean and understandable implementation of a GPT-style language model that can be trained from scratch and extended with modern techniques in future experiments.

## Overview
- Model type: Decoder-only Transformer
- Parameters: ~25M
- Context length: 512
- Vocabulary size: 16,384 (custom BPE tokenizer)
- Attention: GQA flash-Attention (8 query / 4 key-value heads)
- Positional Encoding: RoPE

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
║     GQA flash Attention      ║
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
| n_layers | 8 |
| d_model | 448 |
| n_heads | 8 |
| head_dim | 56 |
| n_kv_heads | 4 |
| ffn_dim | 1152 |
| context_length | 512 |
| vocab_size | 16384 |

## Transformer Block

### Attention
- GQA flash Attention

### Feed Forward Network (SwiGLU)

FFN(x) = W2(SiLU(W1(x)) ⊙ W3(x))

Expansion ratio: 

### Residual Connections

x = x + Attention(x)

x = x + FFN(x)

### Normalization
- RMSNorm

## Parameter Count

| Component        |                                                              Params |
| ---------------- | ------------------------------------------------------------------: |
| Token Embeddings |                                        16384 × 448 = 7,340,032 |
| Attention (GQA)  | [(448×448) + (448×224) + (448×224) + (448×448)] × 8 = 4,816,896 |
| SwiGLU FFN       |             [(448×1152) + (448×1152) + (1152×448)] × 8 = 12,386,304 |
| RMSNorm          |                                               (448 × 2) × 8 = 7,168 |
| Final RMSNorm    |                                                             448 |
| LM Head          |                               weight tied with token embeddings |
| Total            |                                                    **≈ 24.6M** |


## Design Decisions

This model incorporates a subset of modern LLM architectural improvements while remaining small enough to train from scratch on a single NVIDIA T4 GPU.

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

The model is designed to train on a single NVIDIA T4 GPU using Google Colab. Model size, context length, and batch size are selected to balance training speed, memory usage, and model quality within a constrained compute budget.

### Educational Goal

The objective is to bridge the gap between a GPT-2 style transformer and a modern Llama-style architecture while keeping the codebase compact, readable, and suitable for experimentation. Each architectural improvement is implemented from first principles to provide a deeper understanding of how contemporary decoder-only language models are constructed.