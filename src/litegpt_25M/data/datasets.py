from datasets import load_dataset
# import tiktoken
from .tokenizer import encode, EOT_ID
from omegaconf import OmegaConf
import numpy as np
from pathlib import Path
from tqdm import tqdm

cfg = OmegaConf.load("./configs/data/LiteGPT-25M.yaml")
n_vocab = cfg.n_vocab

# encoding = tiktoken.get_encoding(cfg.tokenizer)
# assert encoding.decode(encoding.encode("Hello world")) == "Hello world", (
#     "Tokenizer round-trip failed"
# )
# assert encoding.n_vocab <= np.iinfo(np.uint16).max, "n_vocab is more than uint16"

train_file = Path(cfg.train_bin)
val_file = Path(cfg.val_bin)
split = cfg.train_split

fineweb_target = 300_000_000
code_target = 50_000_000
tinystories_target = 150_000_000

TOTAL_TARGET = fineweb_target + code_target + tinystories_target

# Create output directories
train_file.parent.mkdir(parents=True, exist_ok=True)
val_file.parent.mkdir(parents=True, exist_ok=True)

# Check if pre-tokenized files already exist with sufficient tokens
needs_streaming = True
if train_file.exists() and val_file.exists():
    print("checking if the data is already available...")
    train_tokens = train_file.stat().st_size // 2
    val_tokens = val_file.stat().st_size // 2
    print(f"train_tokens: {train_tokens}")
    print(f"val_tokens: {val_tokens}")
    print(f"total: {train_tokens + val_tokens}")
    if train_tokens + val_tokens >= TOTAL_TARGET:
        print(f"Found token files with {train_tokens + val_tokens:,}/{TOTAL_TARGET:,} tokens. Skipping streaming. :)yay")
        needs_streaming = False
    else:
        print("yep need to download :( ...")

if needs_streaming:
    train_f = open(train_file, "wb")
    val_f = open(val_file, "wb")

    rng = np.random.default_rng(seed=cfg.seed)
    train_count = 0
    val_count = 0

    def write_tokens(text: str):
        tokens = encode(text)
        tokens.append(EOT_ID)
        arr = np.array(tokens, dtype=np.uint16)
        if rng.random() < (1 - split):
            arr.tofile(val_f)
            return 0, len(arr)
        arr.tofile(train_f)
        return len(arr), 0

    global_pbar = tqdm(
        total=TOTAL_TARGET,
        desc="total",
        unit="tok",
        unit_scale=True,
    )

    # Stream fineweb dataset
    print(f"\nStreaming fineweb (target: {fineweb_target:,} tokens)...")
    fineweb = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)

    fineweb_tokens = 0
    pbar = tqdm(total=fineweb_target, desc="fineweb", unit="tok", unit_scale=True)
    for example in fineweb:
        if fineweb_tokens >= fineweb_target:
            break
        text = example.get("text", "")
        if text:
            train_added, val_added = write_tokens(text)
            added = train_added + val_added
            train_count += train_added
            val_count += val_added
            fineweb_tokens += added
            pbar.update(added)
            global_pbar.update(added)
    pbar.close()
    print(f"Fineweb tokens collected: {fineweb_tokens:,}")

    # Stream tinystories dataset
    print(f"\nStreaming tinystories (target: {tinystories_target:,} tokens)...")
    tinystories = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    tinystories_tokens = 0
    pbar = tqdm(
        total=tinystories_target, desc="tinystories", unit="tok", unit_scale=True
    )
    for example in tqdm(tinystories, desc="tinystories"):
        if tinystories_tokens >= tinystories_target:
            break
        text = example.get("text", "")
        if text:
            train_added, val_added = write_tokens(text)
            added = train_added + val_added
            train_count += train_added
            val_count += val_added
            tinystories_tokens += added
            pbar.update(added)
            global_pbar.update(added)
    pbar.close()
    print(f"Tinystories tokens collected: {tinystories_tokens:,}")

    # Stream code dataset
    print(f"\nStreaming code (target: {code_target:,} tokens)...")
    code = load_dataset("bigcode/the-stack-smol", split="train", streaming=True)

    code_tokens = 0
    pbar = tqdm(total=code_target, desc="code", unit="tok", unit_scale=True)
    for example in code:
        if code_tokens >= code_target:
            break
        text = example.get("content", "")
        if text:
            train_added, val_added = write_tokens(text)
            added = train_added + val_added
            train_count += train_added
            val_count += val_added
            code_tokens += added
            pbar.update(added)
            global_pbar.update(added)
    pbar.close()
    print(f"Code tokens collected: {code_tokens:,}")

    train_f.close()
    val_f.close()

    total_tokens = train_count + val_count

    print(f"\nVocab size: {n_vocab:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Training tokens: {train_count:,}")
    print(f"Validation tokens: {val_count:,}")
    
    print(f"train.bin: {train_file.stat().st_size / (1024**3):.2f} GB")
    print(f"val.bin: {val_file.stat().st_size / (1024**3):.2f} GB")

train = np.memmap(train_file, dtype=np.uint16, mode="r")
val = np.memmap(val_file, dtype=np.uint16, mode="r")

print(f"len of training file: {len(train)}")
print(f"len of val file: {len(val)}")
print("=" * 10)
print("min from train: ", train.min())
print("max from train: ", train.max())

print("\nDone!")