# Dataset

## Dataset Overview

| Dataset | Tokens | Purpose |
|----------|---------|---------|
| Shakespeare | 1.1M | Debugging |

## Data Pipeline

Raw Text
   ↓
Cleaning
   ↓
Tokenization
   ↓
Train / Validation Split
   ↓
Binary Storage

## Tokenizer

- Type: tiktoken
- Encoding: gpt2
- Vocabulary size: 50,257

## Cleaning Steps

- Remove duplicates
- Normalize whitespace
- Remove invalid unicode

## Train/Val Split

| Split | Percentage |
|---------|------------|
| Train | 90% |
| Validation | 10% |

## Dataset Statistics

### Raw
- Characters:
- Words:
- Documents:

### Tokenized
- Total Tokens:
- Avg Sequence Length:

## Storage Format

train.bin
val.bin

dtype = uint16

## Future Datasets

- FineWeb
- TinyStories
- OpenWebText