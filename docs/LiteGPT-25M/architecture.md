# Model Architecture

The goal of LiteGPT-25M is not to achieve state-of-the-art performance, but to provide a clean and understandable implementation of a GPT-style language model that can be trained from scratch and extended with modern techniques in future experiments.

## Overview
- Model type: Decoder-only Transformer
- Parameters: ~25M
- Context length: 512
- Vocabulary size: 50,257
- Attention: GQA flash-Attention
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
| d_model | 320 |
| n_heads | 8 |
| head_dim | 40 |
| ffn_dim | 896 |
| context_length | 512 |
| vocab_size | 50257 |

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

| Component        |                                                          Params |
| ---------------- | --------------------------------------------------------------: |
| Token Embeddings |                                        50257 × 320 = 16,082,240 |
| Attention (GQA)  | [(320×320) + (320×160) + (320×160) + (320×320)] × 8 = 3,276,800 |
| SwiGLU FFN       |             [(320×896) + (320×896) + (896×320)] × 8 = 6,881,280 |
| RMSNorm          |                                           (320 × 2) × 8 = 5,120 |
| Final RMSNorm    |                                                             320 |
| LM Head          |                               weight tied with token embeddings |
| Total            |                                                    **≈ 26.25M** |


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