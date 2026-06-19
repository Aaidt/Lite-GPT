import tiktoken
from pathlib import Path
from omegaconf import OmegaConf

cfg = OmegaConf.load("../../configs/data/shakespeare.yaml")
tokenizer = cfg.tokenizer
raw_file = cfg.raw_file

tokenizer = tiktoken.get_encoding(tokenizer)
assert tokenizer.decode(
    tokenizer.encode("Hello world")
) == "Hello world", "Tokenizer round-trip failed"

p = Path(raw_file)
if p.exists() and p.is_file():
    with open(raw_file, "r") as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    print(tokens[:100])
else:
    raise ValueError("Shakespear data not found!!!")