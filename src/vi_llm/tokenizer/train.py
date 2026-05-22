"""BPE tokenizer training on Vietnamese corpus."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tokenizers.normalizers import NFC, NFD, NFKC, NFKD


def _build_normalizer(name: str | None):
    mapping = {"NFC": NFC(), "NFD": NFD(), "NFKC": NFKC(), "NFKD": NFKD()}
    return mapping.get(name or "NFC", NFC())


def _iter_texts(input_dir: str, max_chars: int):
    """Yield raw text strings from Arrow shards up to max_chars total."""
    import datasets

    total = 0
    for shard_dir in sorted(Path(input_dir).rglob("shard_*")):
        ds = datasets.load_from_disk(str(shard_dir))
        for row in ds:
            text = row["text"]
            yield text
            total += len(text)
            if total >= max_chars:
                return


def train_tokenizer(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    vocab_size = cfg.get("vocab_size", 65536)
    min_freq = cfg.get("min_frequency", 2)
    special_tokens = cfg.get("special_tokens", ["<s>", "</s>", "<pad>", "<unk>"])
    max_chars = cfg.get("max_train_chars", 10_000_000_000)
    input_dir = cfg["input_dir"]
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = _build_normalizer(cfg.get("normalizer"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_tokens,
        show_progress=True,
    )

    tokenizer.train_from_iterator(
        _iter_texts(input_dir, max_chars),
        trainer=trainer,
    )

    tokenizer.save(str(output_dir / "tokenizer.json"))
    _save_hf_compat(tokenizer, special_tokens, output_dir)
    print(f"Tokenizer saved to {output_dir}  (vocab size: {tokenizer.get_vocab_size()})")

    hf_repo = cfg.get("hf_repo")
    if hf_repo:
        push_to_hub(str(output_dir), hf_repo)


def _save_hf_compat(tokenizer, special_tokens: list[str], output_dir: Path) -> None:
    """Write tokenizer_config.json for HF AutoTokenizer compatibility."""
    import json

    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": special_tokens[0] if len(special_tokens) > 0 else "<s>",
        "eos_token": special_tokens[1] if len(special_tokens) > 1 else "</s>",
        "pad_token": special_tokens[2] if len(special_tokens) > 2 else "<pad>",
        "unk_token": special_tokens[3] if len(special_tokens) > 3 else "<unk>",
        "model_max_length": 2048,
        "padding_side": "right",
    }
    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def push_to_hub(tokenizer_dir: str, repo_id: str | None = None) -> None:
    from huggingface_hub import HfApi

    if repo_id is None:
        repo_id = os.environ.get("HF_TOKENIZER_REPO")
    if not repo_id:
        raise ValueError("Set hf_repo in tokenizer.yaml or HF_TOKENIZER_REPO env var.")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=tokenizer_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Tokenizer pushed to https://huggingface.co/{repo_id}")
