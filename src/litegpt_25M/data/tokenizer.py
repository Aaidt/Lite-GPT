from datasets import load_dataset
from tqdm import tqdm
from omegaconf import OmegaConf
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from pathlib import Path

cfg = OmegaConf.load("./configs/data/LiteGPT-25M.yaml")
corpus_file = cfg.corpus_file
tokenizer_path = cfg.tokenizer_path

# create a training corpus having 60% finewweb + 30% tinystories + 10% the stack code
target_words = 50_000_000
fineweb_target = 30_000_000
tinystories_target = 15_000_000
code_target = 5_000_000

global_pbar = tqdm(
    total=target_words,
    desc="total",
    unit="char",
    unit_scale=True,
)


streaming = True
if Path(tokenizer_path).exists():
    print(
        f"Tokenizer already exists. Skipping corpus creation and training."
    )
    streaming = False
elif Path(corpus_file).exists() and Path(corpus_file).stat().st_size > 0:
    print(f"Corpus file already exists and is not empty. Skipping corpus creation.")
    streaming = False

if streaming:
    corpus_f = open(corpus_file, "wb")
    print(f"Creating corpus file {corpus_file}...")
    # stream all the datasets and store it in a txt file
    print(f"\nStreaming fineweb (target: {fineweb_target:,} tokens)...")
    fineweb = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
    )

    fineweb_tokens = 0
    pbar = tqdm(total=fineweb_target, desc="fineweb", unit="tok", unit_scale=True)
    for example in fineweb:
        if fineweb_tokens >= fineweb_target:
            break
        text = example.get("text", "")
        if text:
            corpus_f.write((text + "\n").encode("utf-8"))
            added = len(text)
            fineweb_tokens += added
            pbar.update(added)
            global_pbar.update(added)
    pbar.close()
    print(f"Fineweb tokens collected: {fineweb_tokens:,}")

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
            corpus_f.write((text + "\n").encode("utf-8"))
            added = len(text)
            tinystories_tokens += added
            pbar.update(added)
            global_pbar.update(added)
    pbar.close()
    print(f"Tinystories tokens collected: {tinystories_tokens:,}")

    print(f"\nStreaming code (target: {code_target:,} tokens)...")
    code = load_dataset("bigcode/the-stack-smol", split="train", streaming=True)

    code_tokens = 0
    pbar = tqdm(total=code_target, desc="code", unit="tok", unit_scale=True)
    for example in code:
        if code_tokens >= code_target:
            break
        text = example.get("content", "")
        if text:
            corpus_f.write((text + "\n").encode("utf-8"))
            added = len(text)
            code_tokens += added
            pbar.update(added)
            global_pbar.update(added)
    pbar.close()
    print(f"Code tokens collected: {code_tokens:,}")

    corpus_f.close()

tokenizer_path = Path(tokenizer_path)


def test_tokenizer(tokenizer: Tokenizer, text: str) -> None:
    ids = tokenizer.encode(text).ids
    print(f"Original text: {text}")
    print(f"Token IDs: {ids}")
    decoded = tokenizer.decode(ids)
    print(f"Decoded text: {decoded}")
    assert decoded == text, "Decoded text does not match original text"


if tokenizer_path.exists():
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    EOT_ID = tokenizer.token_to_id("<|endoftext|>")

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text).ids

else:
    # train a byte-level BPE on this corpus
    tokenizer = Tokenizer(models.BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=16384,
        special_tokens=["<|endoftext|>"],
    )

    tokenizer.train(
        [corpus_file],
        trainer=trainer,
    )

    tokenizer.save(str(tokenizer_path))
    test_tokenizer(tokenizer, "Hello, world!")
