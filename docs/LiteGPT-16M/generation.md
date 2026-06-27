# Generation

The LiteGPT-16M model is a GPT-2 style decoder-only transformer trained on the Tiny Shakespeare dataset. It generates Shakespeare-like text.

## Generation Settings

| Parameter | Default |
|-----------|---------|
| Temperature | 1.0 |
| Top-k | None |
| Max Tokens | 100 |

## Example Prompts

The model can be prompted with passages from Shakespeare to generate continuations:

- "To be, or not to be,"
- "ROMEO: But, soft! what light through yonder window breaks?"
- "All the world's a stage,"
- "HAMLET: To sleep, perchance to dream—"

## Running Generation

The 16M model does not have a dedicated evaluation script. You can write a simple generation loop using the model's forward pass:

```python
import torch
from src.litegpt_16M.model.model import LiteGPT
from tokenizers import Tokenizer

model = LiteGPT()
model.load_state_dict(torch.load("checkpoint.pt"))
model.eval()

# Encode prompt with GPT-2 tokenizer (tiktoken)
tokenizer = Tokenizer.from_pretrained("gpt2")
prompt = "To be, or not to be,"
tokens = torch.tensor(tokenizer.encode(prompt).ids).unsqueeze(0)

# Greedy generation loop
with torch.no_grad():
    for _ in range(100):
        logits, _ = model(tokens, None)
        next_token = logits[0, -1].argmax()
        tokens = torch.cat([tokens, next_token.view(1, -1)], dim=1)

print(tokenizer.decode(tokens[0].tolist()))
```
