import tiktoken
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

cfg = OmegaConf.load("../../configs/data/shakespeare.yaml")

encoder = tiktoken.get_encoding(cfg.tokenizer)
assert encoder.decode(
    encoder.encode("Hello world")
) == "Hello world", "Tokenizer round-trip failed"

assert encoder.n_vocab <= np.iinfo(np.uint16).max, "n_vocab is more than uint16"

p = Path(cfg.raw_file)
train_file = Path(cfg.train_bin)
val_file = Path(cfg.val_bin) 

if p.exists() and p.is_file():
    with open(cfg.raw_file, "r") as f:
        text = f.read()

    tokens = encoder.encode(text)
    # print(tokens[:100])   
    tokens = np.array(tokens, dtype=np.uint16)

    train_tokens = tokens[: int(0.9 * len(tokens))]
    val_tokens = tokens[int(0.9 * len(tokens)): ]

    print(f"Total tokens: {len(tokens):,}")
    print(f"Training tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")

    if not train_file.exists() and not val_file.exists():
        train_file.parent.mkdir(parents=True, exist_ok=True)
        val_file.parent.mkdir(parents=True, exist_ok=True)
        # train_file.touch()
        # val_file.touch()

    train_tokens.tofile(cfg.train_bin)
    val_tokens.tofile(cfg.val_bin)

    print(f"\ntrain.bin: {train_file.stat().st_size / 1024:.2f} KB")
    print(f"val.bin: {val_file.stat().st_size / 1024:.2f} KB")

else:
    raise ValueError("Shakespear data not found!!!")