from datasets import load_dataset
from tqdm import tqdm
from omegaconf import OmegaConf
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from pathlib import Path

cfg = OmegaConf.load("./configs/data/LiteGPT-25M.yaml")
corpus_file = cfg.corpus_file
tokenizer_path = Path(cfg.tokenizer_path)

# create a training corpus having 67% fineweb + 33% tinystories
target_words = 45_000_000
fineweb_target = 30_000_000
tinystories_target = 15_000_000

global_pbar = tqdm(
    total=target_words,
    desc="total",
    unit="char",
    unit_scale=True,
)


def validate_tokenizer(path: Path) -> bool:
    try:
        tokenizer = Tokenizer.from_file(str(path))
        sample = "Hello, world!"
        ids = tokenizer.encode(sample).ids
        if tokenizer.decode(ids) != sample:
            print("Tokenizer validation failed: round-trip mismatch.")
            return False
        if tokenizer.token_to_id("<|endoftext|>") is None:
            print("Tokenizer validation failed: missing <|endoftext|> token.")
            return False
        if tokenizer.get_vocab_size() == 0:
            print("Tokenizer validation failed: empty vocabulary.")
            return False
        return True
    except Exception as exc:
        print(f"Tokenizer validation failed: {exc}")
        return False


streaming = True
if tokenizer_path.exists():
    print("Tokenizer already exists. Validating...")
    if validate_tokenizer(tokenizer_path):
        print("Tokenizer is valid. Skipping corpus creation and training.")
        streaming = False
    else:
        print("Tokenizer exists but is invalid. Recreating tokenizer.")
        tokenizer_path.unlink(missing_ok=True)

if streaming and Path(corpus_file).exists() and Path(corpus_file).stat().st_size > 0:
    print(f"Corpus file already exists and is not empty. Skipping corpus creation.")
    streaming = False

if streaming:
    corpus_path = Path(corpus_file)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Creating corpus file {corpus_file}...")
    with open(corpus_path, "wb") as corpus_f:
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
