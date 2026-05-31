"""Stage 5 — Tokenise corpus and pack into fixed-length CLM sequences."""

from __future__ import annotations

from pathlib import Path

# ── Parallel worker ────────────────────────────────────────────────────────────

_worker_tokenizer = None


def _init_tokenizer_worker(tokenizer_dir: str) -> None:
    global _worker_tokenizer
    from transformers import PreTrainedTokenizerFast
    _worker_tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    # We chunk manually, so disable the per-encode length check.
    _worker_tokenizer.model_max_length = int(1e30)


def _tokenize_shard(shard_dir: str) -> "np.ndarray":
    """Tokenise all docs in one shard; return a flat int32 token array with EOS appended per doc."""
    import numpy as np
    import datasets as hf_datasets
    ds = hf_datasets.load_from_disk(shard_dir)
    eos_id = _worker_tokenizer.eos_token_id
    buf: list[int] = []
    for text in ds["text"]:
        buf.extend(_worker_tokenizer.encode(text))
        buf.append(eos_id)
    return np.array(buf, dtype=np.int32)


# ── Public entry point ─────────────────────────────────────────────────────────

def pack_dataset(
    tokenizer_dir: str,
    input_dir: str,
    output_dir: str,
    seq_length: int = 2048,
    shard_size: int = 10_000,
    num_workers: int = 4,
) -> None:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np
    import datasets as hf_datasets
    from tqdm import tqdm

    from vi_llm.dataprep.utils import is_done, mark_done

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if is_done(output_path):
        print(f"Skipping pack — already done at {output_path}")
        return

    shard_dirs = sorted(str(p) for p in input_path.rglob("shard_*"))
    if not shard_dirs:
        print(f"No shards found in {input_dir}.")
        return

    print(f"Packing {len(shard_dirs)} shards  seq_length={seq_length}  workers={num_workers}")

    executor = None
    if num_workers == 1:
        _init_tokenizer_worker(tokenizer_dir)
        shard_results = map(_tokenize_shard, shard_dirs)
    else:
        executor = ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=multiprocessing.get_context("fork"),
            initializer=_init_tokenizer_worker,
            initargs=(tokenizer_dir,),
        )
        shard_results = executor.map(_tokenize_shard, shard_dirs)

    remainder = np.empty(0, dtype=np.int32)
    out_shard_idx = 0
    seq_buffer: list[list[int]] = []
    total_tokens = total_seqs = 0

    try:
        with tqdm(total=len(shard_dirs), desc="Packing shards", unit="shard") as pbar:
            for token_arr in shard_results:
                total_tokens += len(token_arr)
                combined = np.concatenate([remainder, token_arr])
                n_full = len(combined) // seq_length
                for i in range(n_full):
                    seq_buffer.append(combined[i * seq_length : (i + 1) * seq_length].tolist())
                    total_seqs += 1
                    if len(seq_buffer) >= shard_size:
                        _flush_shard(seq_buffer, output_path, out_shard_idx, hf_datasets)
                        out_shard_idx += 1
                        seq_buffer = []
                remainder = combined[n_full * seq_length :]
                pbar.set_postfix(seqs=total_seqs, tokens=f"{total_tokens / 1e6:.1f}M")
                pbar.update(1)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if seq_buffer:
        _flush_shard(seq_buffer, output_path, out_shard_idx, hf_datasets)

    mark_done(output_path)
    print(
        f"\nPacked: {total_seqs:,} sequences × {seq_length} tokens"
        f" = {total_seqs * seq_length:,} total tokens  →  {output_path}"
    )


def _flush_shard(seqs: list[list[int]], output_dir: Path, shard_idx: int, hf_datasets) -> None:
    shard_path = output_dir / f"shard_{shard_idx:05d}"
    hf_datasets.Dataset.from_dict({"input_ids": seqs}).save_to_disk(str(shard_path))
