<div align="center">

```
██╗     ██╗████████╗███████╗     ██████╗ ██████╗ ████████╗
██║     ██║╚══██╔══╝██╔════╝    ██╔════╝ ██╔══██╗╚══██╔══╝
██║     ██║   ██║   █████╗      ██║  ███╗██████╔╝   ██║   
██║     ██║   ██║   ██╔══╝      ██║   ██║██╔═══╝    ██║   
███████╗██║   ██║   ███████╗    ╚██████╔╝██║        ██║   
```

> *A clean, educational decoder-only Transformer language model (~25M parameters) trained from scratch on a single NVIDIA A5000 GPU.*

---

</div>

### Architecture

| Configuration | Value |
|---|---|
| Model type | Decoder-only Transformer |
| Parameters | ~24.6M |
| Layers | 8 |
| d_model | 448 |
| Attention heads | 8 query / 4 key-value (GQA, head_dim=56) |
| FFN dimension | 1152 (SwiGLU) |
| Context length | 512 |
| Vocabulary | 16,384 (custom BPE tokenizer) |

The model incorporates modern LLM improvements over GPT-2:
- **RoPE** positional encodings
- **Grouped Query Attention (GQA)** with FlashAttention
- **SwiGLU** feed-forward networks
- **RMSNorm** with pre-norm residual connections
- **Weight tying** between token embeddings and LM head

### Datasets

Trained on ~1B tokens (custom BPE tokenizer, vocab 16,384):

| Dataset | Tokens | Weight |
|---|---|---|
| FineWeb | 300M | 60% |
| TinyStories | 150M | 30% |
| The Stack Smol | 50M | 10% |

Data is tokenized with a custom ByteLevel BPE tokenizer (vocab size 16,384), stored as `uint16` arrays, and split 90/10 train/validation.

### Training

| Hyperparameter | Value |
|---|---|
| Effective batch size | 128 |
| Sequence length | 512 |
| Learning rate | 6e-4 → 6e-5 (cosine decay) |
| Warmup steps | 4000 |
| Max steps | 40000 |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Precision | BF16/FP16 mixed |
| Tokens seen | ~1B |

Trained on a **NVIDIA A5000 GPU** (Runpod) using Cross Entropy Loss with cosine LR schedule and periodic checkpointing.
<!-- 
### Results

| Metric | Value |
|---|---|
| Train Loss | 2.88 |
| Val Loss | 5.99 |
| Perplexity | 397.7 | -->