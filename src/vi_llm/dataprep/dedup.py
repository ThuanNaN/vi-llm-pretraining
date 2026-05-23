"""Stage 3 — MinHash LSH near-duplicate removal."""

from __future__ import annotations

from pathlib import Path


def _shingles(text: str, n: int = 5) -> set[str]:
    """Return character n-gram shingles. Falls back to {text} when len(text) < n."""
    if len(text) <= n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _make_minhash(text: str, num_perm: int = 128):
    from datasketch import MinHash
    m = MinHash(num_perm=num_perm)
    for shingle in _shingles(text):
        m.update(shingle.encode("utf-8"))
    return m


def dedup_dataset(
    input_dir: str,
    output_dir: str,
    threshold: float = 0.85,
    num_perm: int = 128,
) -> None:
    import datasets as hf_datasets
    from datasketch import MinHashLSH

    from vi_llm.dataprep.utils import is_done, mark_done

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if is_done(output_path):
        print(f"Skipping dedup — already done at {output_path}")
        return

    shard_dirs = sorted(input_path.rglob("shard_*"))
    if not shard_dirs:
        print(f"No shards found in {input_dir} — run make clean-data first.")
        return

    print(f"Deduplicating {len(shard_dirs)} shards  threshold={threshold}  num_perm={num_perm}")

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    shard_size = 10_000
    shard_idx = 0
    buffer: list[str] = []
    total_in = total_out = doc_id = 0

    for shard_dir in shard_dirs:
        ds = hf_datasets.load_from_disk(str(shard_dir))
        for text in ds["text"]:
            total_in += 1
            minhash = _make_minhash(text, num_perm)
            key = f"d{doc_id}"
            doc_id += 1

            if not lsh.query(minhash):
                lsh.insert(key, minhash)
                buffer.append(text)
                total_out += 1

                if len(buffer) >= shard_size:
                    _flush_shard(buffer, output_path, shard_idx, hf_datasets)
                    shard_idx += 1
                    buffer = []

            if total_in % 10_000 == 0:
                pct = total_out / total_in * 100
                print(f"  {total_in} processed  {total_out} kept ({pct:.1f}%)", end="\r")

    if buffer:
        _flush_shard(buffer, output_path, shard_idx, hf_datasets)

    mark_done(output_path)
    pct = total_out / total_in * 100 if total_in else 0
    print(f"\nDedup: {total_out}/{total_in} kept ({pct:.1f}%)  →  {output_path}")


def _flush_shard(texts: list[str], output_dir: Path, shard_idx: int, hf_datasets) -> None:
    shard_path = output_dir / f"shard_{shard_idx:05d}"
    hf_datasets.Dataset.from_dict({"text": texts}).save_to_disk(str(shard_path))
