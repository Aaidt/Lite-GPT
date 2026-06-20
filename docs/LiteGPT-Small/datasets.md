# Dataset

The Tiny Shakespeare dataset is the primary dataset used for training LiteGPT-Small. It provides a small but well-structured corpus for testing tokenization, data loading, training, and text generation. The dataset contains approximately 338k GPT-2 tokens and is intended for experimentation rather than large-scale pretraining.

## Dataset Statistics

| Metric | Value |
|----------|---------|
| Characters | 1,115,394 |
| GPT-2 Tokens | ~338,000 |
| Vocabulary Used | ~11,706 |
| Full GPT-2 Vocabulary | 50,257 |

## Data Pipeline

```text
shakespeare.txt
       │
       ▼
┌──────────────────┐
│ GPT-2 Tokenizer  │
│ (tiktoken)       │
└────────┬─────────┘
         │
         ▼
   Token IDs
         │
         ▼
┌──────────────────┐
│ 90/10 Split      │
│ Train / Val      │
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
train.bin  val.bin
(uint16)   (uint16)
```

## Tokenizer

- Type: tiktoken
- Encoding: gpt2
- Vocabulary size: 50,257

## Train/Val Split

| Split | Percentage |
|---------|------------|
| Train | 90% |
| Validation | 10% |

## Storage Format

train.bin

val.bin

dtype = uint16

## Future Datasets

- FineWeb
- TinyStories
- OpenWebText