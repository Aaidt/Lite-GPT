# Training

## Hardware

- GPU: T4 16GB
- Platform: Google Colab

## Hyperparameters
| Parameter             | Value |
| --------------------- | ----- |
| Batch Size            | 64    |
| Gradient Accumulation | 2     |
| Effective Batch Size  | 128   |
| Sequence Length       | 512   |
| Learning Rate         | 6e-4  |
| Min Learning Rate     | 6e-5  |
| Weight Decay          | 0.1   |
| Warmup Steps          | 1550  |
| Max Steps             | 15500 |
| Eval Interval         | 500   |
| Eval Iters            | 100   |
| Grad Clip             | 1.0   |

## Tokens Seen
```text
64 × 512 x 15500
= 507,904,000 tokens

≈ 507M tokens.
```

## Optimizer

AdamW

β1 = 0.9
β2 = 0.95

## Learning Rate Schedule

Cosine Decay

Warmup: 0 → 6e-4 (125 optimizer steps / 1000 raw iterations)
Decay: 6e-4 → 6e-5 (1125 optimizer steps)

## Loss Function

Cross Entropy Loss

## Mixed Precision

- BF16 / FP16

## Checkpointing

Save:
- model state
- optimizer state
- scheduler state
- training step

Frequency:
- every N steps

## Dataset mix

| Dataset     | Weight |
| ----------- | ------ |
| FineWeb     | 60%    |
| TinyStories | 30%    |
| Code        | 10%    |


## Evaluation

### Metrics

- Validation Loss
- Perplexity

Perplexity = exp(loss)

## Generation Settings

| Parameter | Value |
|------------|---------|
| Temperature | 1.0 |
| Top-k | None |
| Max Tokens | 100 |

<!-- ## Experiments

### Experiment 1

Config:
Results:
Observations:

### Experiment 2

Config:
Results:
Observations: -->

<!-- ## Expected range

| Metric     | Good      |
| ---------- | --------- |
| Train Loss | <2.5      |
| Val Loss   | <2.5      |
| Perplexity | <20       | -->

## Final Results

| Metric | Value |
|---------|---------|
| Train Loss | 2.9977498054504395 |
| Val Loss | 2.9335184621810915 |
| Perplexity | 18.793639012453678 |