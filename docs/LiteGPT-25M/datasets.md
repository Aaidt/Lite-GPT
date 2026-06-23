# Dataset

LiteGPT-25M is trained on a curated mixture of web text, source code, and synthetic stories. The dataset is designed to provide broad language coverage while remaining practical to preprocess and train on a single NVIDIA T4 GPU.

The final corpus contains approximately **500 million GPT-2 tokens** distributed across three datasets.

## Dataset Composition

| Dataset        |   Tokens | Percentage | Purpose                                             |
| -------------- | -------: | ---------: | --------------------------------------------------- |
| FineWeb        |     300M |        60% | General web knowledge and educational content       |
| TinyStories    |     150M |        30% | Narrative structure, grammar, and language modeling |
| The Stack Smol |      50M |        10% | Source code and programming patterns                |
| **Total**      | **500M** |   **100%** |                                                     |

## Dataset Statistics

| Metric           | Value            |
| ---------------- | ---------------- |
| Total Tokens     | ~500M            |
| Tokenizer        | GPT-2 (tiktoken) |
| Vocabulary Size  | 50,257           |
| Context Length   | 512              |
| Train Split      | 90%              |
| Validation Split | 10%              |
| Storage Format   | uint16           |

## Data Pipeline

```text
FineWeb (300M)
TinyStories (150M)
The Stack Smol (50M)
        │
        ▼
┌──────────────────┐
│ Dataset Mixing   │
└────────┬─────────┘
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

* Type: tiktoken
* Encoding: gpt2
* Vocabulary size: 50,257
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

The dataset mixture balances three complementary sources of information:

* **FineWeb** provides broad factual knowledge, educational content, and general language understanding.
* **TinyStories** improves grammar, coherence, narrative structure, and basic reasoning.
* **The Stack Smol** introduces programming syntax, code structure, and software engineering concepts.

This combination creates a diverse corpus that is substantially richer than any individual dataset while remaining computationally feasible for small-scale pretraining experiments.
