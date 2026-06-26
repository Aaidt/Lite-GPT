# Dataset

LiteGPT-25M is trained on a curated mixture of web text and synthetic stories. The dataset is designed to provide broad language coverage while remaining practical to preprocess and train on a single NVIDIA T4 GPU.

The final corpus contains approximately **500 million tokens** (using a custom ByteLevel BPE tokenizer with 16,384 vocabulary) distributed across two datasets.

## Dataset Composition

| Dataset        |   Tokens | Percentage | Purpose                                             |
| -------------- | -------: | ---------: | --------------------------------------------------- |
| FineWeb        |     300M |        60% | General web knowledge and educational content       |
| TinyStories    |     200M |        40% | Narrative structure, grammar, and language modeling |
| **Total**      | **500M** |   **100%** |                                                     |

## Dataset Statistics

| Metric           | Value            |
| ---------------- | ---------------- |
| Total Tokens     | ~500M            |
| Tokenizer        | Custom ByteLevel BPE |
| Vocabulary Size  | 16,384               |
| Context Length   | 512              |
| Train Split      | 90%              |
| Validation Split | 10%              |
| Storage Format   | uint16           |

## Data Pipeline

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

## Tokenizer

* Type: ByteLevel BPE (trained on the corpus)
* Vocabulary size: 16,384
* Special tokens: `<|endoftext|>`
* End-of-text token appended after every document
* Tokens stored as uint16 for efficient disk usage

## Train/Validation Split

| Split      | Percentage |
| ---------- | ---------- |
| Train      | 90%        |
| Validation | 10%        |

Documents are assigned to the training or validation set using a random split during preprocessing.

## Storage Format

```text
train.bin
val.bin

dtype = uint16
```

Token IDs are streamed directly to disk during preprocessing and loaded through memory mapping during training.

## Why This Dataset Mix?

The dataset mixture balances two complementary sources of information:

* **FineWeb** provides broad factual knowledge, educational content, and general language understanding.
* **TinyStories** improves grammar, coherence, narrative structure, and basic reasoning.

This combination creates a diverse corpus that is substantially richer than any individual dataset while remaining computationally feasible for small-scale pretraining experiments.
