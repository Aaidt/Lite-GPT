# Training

## Hardware

- GPU: T4 16GB
- Platform: Google Colab

## Hyperparameters

| Parameter | Value |
|------------|---------|
| Batch Size | 16 |
| Sequence Length | 256 |
| Learning Rate | 3e-4 |
| Weight Decay | 0.1 |
| Warmup Steps | 2000 |
| Max Steps | 40000 |

## Optimizer

AdamW

β1 = 0.9
β2 = 0.95

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

## Final Results

| Metric | Value |
|---------|---------|
| Train Loss | 2.878491520881653 |
| Val Loss | 5.985600624084473 |
| Perplexity | 397.6612944866264 |