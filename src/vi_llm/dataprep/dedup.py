"""Stage 3 — MinHash LSH near-duplicate removal."""

from __future__ import annotations

from pathlib import Path
import numpy as np



def _shingles(text: str, n: int = 5) -> set[str]:
    """Return character n-gram shingles. Falls back to {text} when len(text) < n."""
    if len(text) <= n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _hashvalues(text: str, num_perm: int) -> "np.ndarray":
    """Return MinHash hashvalues (uint32 array of shape num_perm) for one document.

    Returns only the raw numpy array — not a MinHash object — so the worker
    can send a single stacked array over IPC instead of thousands of Python
    objects with large _hash_ranges tuples attached.
    """
    from datasketch import MinHash
    m = MinHash(num_perm=num_perm)
    for shingle in _shingles(text):
        m.update(shingle.encode("utf-8"))
    return m.hashvalues.copy()


# ── Parallel worker ────────────────────────────────────────────────────────────

def _minhash_shard(args: tuple[str, int]) -> tuple[list[str], "np.ndarray", int]:
    """Load one shard and return (texts, hashvalues_array, n_docs).

    hashvalues_array has shape (n_docs, num_perm) dtype uint32.
    Sending one ndarray over IPC is ~100x cheaper than pickling MinHash objects.
    """
    import numpy as np
    import datasets as hf_datasets
    shard_dir, num_perm = args
    texts: list[str] = hf_datasets.load_from_disk(shard_dir)["text"]
    arr = np.stack([_hashvalues(t, num_perm) for t in texts])
    return texts, arr, len(texts)


# ── Public entry points ────────────────────────────────────────────────────────

def dedup_dataset(config_path: str) -> None:
    import datasets as hf_datasets
    import yaml
    from datasketch import MinHashLSH
    from tqdm import tqdm

    from vi_llm.dataprep.utils import is_done, mark_done

    with open(config_path) as f:
        top = yaml.safe_load(f)
    cfg = top.get("dedup", top)

    input_path = Path(cfg["input_dir"])
    output_path = Path(cfg["output_dir"])
    threshold = cfg.get("threshold", 0.85)
    num_perm = cfg.get("num_perm", 128)
    shard_size = cfg.get("shard_size", 10_000)
    num_workers = cfg.get("num_workers", 1)

    if is_done(output_path):
        print(f"Skipping dedup — already done at {output_path}")
        return

    shard_dirs = sorted(str(p) for p in input_path.rglob("shard_*"))
    if not shard_dirs:
        print(f"No shards found in {input_path} — run make clean-data first.")
        return

    print(
        f"Deduplicating {len(shard_dirs)} shards  "
        f"threshold={threshold}  num_perm={num_perm}  workers={num_workers}"
    )

    args = [(d, num_perm) for d in shard_dirs]

    executor = None
    if num_workers == 1:
        shard_results = map(_minhash_shard, args)
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        executor = ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=multiprocessing.get_context("fork"),
        )
        shard_results = executor.map(_minhash_shard, args)

    from datasketch import MinHash as _MinHash
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    shard_idx = 0
    buffer: list[str] = []
    total_in = total_out = doc_id = 0

    try:
        with tqdm(total=len(shard_dirs), desc="Deduplicating shards", unit="shard") as pbar:
            for texts, hashvalues_arr, n_in in shard_results:
                total_in += n_in
                for text, hv in zip(texts, hashvalues_arr):
                    key = f"d{doc_id}"
                    doc_id += 1
                    m = _MinHash(num_perm=num_perm)
                    m.hashvalues = hv
                    if not lsh.query(m):
                        lsh.insert(key, m)
                        buffer.append(text)
                        total_out += 1
                        if len(buffer) >= shard_size:
                            _flush_shard(buffer, output_path, shard_idx, hf_datasets)
                            shard_idx += 1
                            buffer = []
                pct = total_out / total_in * 100 if total_in else 0
                pbar.set_postfix(kept=total_out, total=total_in, pct=f"{pct:.1f}%")
                pbar.update(1)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if buffer:
        _flush_shard(buffer, output_path, shard_idx, hf_datasets)

    mark_done(output_path)
    pct = total_out / total_in * 100 if total_in else 0
    print(f"\nDedup: {total_out}/{total_in} kept ({pct:.1f}%)  →  {output_path}")


def _flush_shard(texts: list[str], output_dir: Path, shard_idx: int, hf_datasets) -> None:
    shard_path = output_dir / f"shard_{shard_idx:05d}"
    hf_datasets.Dataset.from_dict({"text": texts}).save_to_disk(str(shard_path))
