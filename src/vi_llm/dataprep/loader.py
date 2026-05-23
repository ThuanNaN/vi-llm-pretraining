"""Stage 1 — Download datasets from HuggingFace Hub and save as Arrow shards.

Reads parquet files directly via pyarrow + HfFileSystem to avoid the
large_string → string cast error that occurs in the datasets streaming path.

Row → text normalisation is handled by vi_llm.dataprep.converters (priority 1)
or by text_columns in configs/datasets.yaml (priority 2).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from vi_llm.dataprep.converters import get as get_converter
from vi_llm.dataprep.utils import is_done, mark_done


def download_all(config_path: str) -> None:
    import datasets as hf_datasets

    with open(config_path) as f:
        top = yaml.safe_load(f)
    cfg = top.get("download", top)

    output_dir = Path(cfg["output_dir"])
    shard_size = cfg.get("shard_size", 10_000)

    for entry in cfg["datasets"]:
        _download_dataset(entry, output_dir, shard_size, hf_datasets)


def _find_parquet_files(fs, dataset_id: str, name: str | None, split: str) -> list[str]:
    """Return sorted parquet paths on HF Hub for the given config/split."""
    base = f"datasets/{dataset_id}"

    patterns: list[str] = []
    if name:
        patterns += [
            f"{base}/data/{name}.parquet",
            f"{base}/data/{name}-*.parquet",
            f"{base}/{name}/{split}-*.parquet",
            f"{base}/{name}/data/{split}-*.parquet",
        ]
    patterns += [
        f"{base}/data/{split}-*.parquet",
        f"{base}/{split}-*.parquet",
    ]

    for pattern in patterns:
        if "*" not in pattern:
            try:
                if fs.exists(pattern):
                    return [pattern]
            except Exception:
                pass
        else:
            found = sorted(fs.glob(pattern))
            if found:
                return found

    raise FileNotFoundError(
        f"No parquet files found for dataset='{dataset_id}', name='{name}', split='{split}'. "
        f"Check available configs with: datasets.get_dataset_config_names('{dataset_id}')"
    )


def _download_dataset(entry: dict, output_dir: Path, shard_size: int, hf_datasets) -> None:
    import pyarrow.fs as pa_fs
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    dataset_id = entry["id"]
    name = entry.get("name")
    split = entry.get("split", "train")
    text_columns: list[str] | None = entry.get("text_columns")
    max_docs = entry.get("max_docs")

    converter = get_converter(dataset_id)
    if converter is None and not text_columns:
        raise ValueError(
            f"No converter registered for '{dataset_id}' and no text_columns in config. "
            f"Add a converter to vi_llm/data/converters.py or set text_columns in configs/datasets.yaml."
        )

    safe_name = dataset_id.replace("/", "__")
    dataset_dir = output_dir / safe_name
    if is_done(dataset_dir):
        print(f"Skipping {dataset_id} — already downloaded at {dataset_dir}")
        return

    print(f"Downloading {dataset_id} (split={split}, name={name}) ...")

    fs = HfFileSystem()
    pa_filesystem = pa_fs.PyFileSystem(pa_fs.FSSpecHandler(fs))
    files = _find_parquet_files(fs, dataset_id, name, split)

    # Read only the needed columns when falling back to text_columns config.
    # With a registered converter we read all columns (it decides what to use).
    read_columns = text_columns if converter is None else None

    shard_idx = 0
    buffer: list[str] = []
    total = 0
    done = False

    for file_path in files:
        if done:
            break
        pf = pq.ParquetFile(file_path, filesystem=pa_filesystem)
        for batch in pf.iter_batches(batch_size=500, columns=read_columns):
            if done:
                break
            col_arrays = {col: batch.column(col) for col in batch.schema.names}
            for i in range(batch.num_rows):
                if max_docs is not None and total >= max_docs:
                    done = True
                    break

                row = {col: arr[i].as_py() for col, arr in col_arrays.items()}

                if converter is not None:
                    text = converter(row)
                else:
                    parts = [str(row[col]) for col in (text_columns or []) if row.get(col)]
                    text = "\n\n".join(parts) or None

                if not text or not text.strip():
                    continue

                buffer.append(text)
                total += 1

                if len(buffer) >= shard_size:
                    _flush_shard(buffer, dataset_dir, shard_idx, hf_datasets)
                    shard_idx += 1
                    buffer = []

    if buffer:
        _flush_shard(buffer, dataset_dir, shard_idx, hf_datasets)

    mark_done(dataset_dir)
    num_shards = shard_idx + (1 if buffer else 0)
    print(f"  Saved {total} documents to {dataset_dir}  ({num_shards} shards)")


def _flush_shard(texts: list[str], dataset_dir: Path, shard_idx: int, hf_datasets) -> None:
    shard_path = dataset_dir / f"shard_{shard_idx:05d}"
    hf_datasets.Dataset.from_dict({"text": texts}).save_to_disk(str(shard_path))
