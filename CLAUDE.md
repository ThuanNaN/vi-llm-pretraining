# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

End-to-end pre-training pipeline for a Vietnamese causal LLM (1–7B, LLaMA architecture). The pipeline runs in six numbered stages; each stage reads from the previous stage's output directory and writes Arrow shards to disk.

## Setup

```bash
pip install uv
uv pip install -e .
uv pip install -e ".[cuda]"   # adds DeepSpeed (Linux/CUDA only)
```

Copy `.env.example` to `.env` and set `HF_TOKEN` for Hub access.

## Common commands

```bash
# Run all data pipeline stages (1–5)
make pipeline

# Individual pipeline stages
make download       # stage 1: HF Hub → data/raw/
make clean-data     # stage 2: data/raw/ → data/cleaned/
make dedup          # stage 3: data/cleaned/ → data/deduped/
make tokenizer      # stage 4: train BPE tokenizer → artifacts/tokenizer/
make tokenize       # stage 5: data/deduped/ → data/packed/

# Training
make train-1b            # single GPU / MPS
make train-7b
make train-1b-fsdp       # multi-GPU FSDP (CUDA only)
make train-7b-fsdp

# Hub push
make push-tokenizer
make push-checkpoint

# Tests
make test
pytest tests/ -v
pytest tests/test_cleaner.py -v   # run a single test file
```

## Architecture

### Data flow

```
data/raw/{dataset_id}__*/shard_NNNNN/
  → data/cleaned/shard_NNNNN/
  → data/deduped/shard_NNNNN/
  → data/packed/shard_NNNNN/   (input_ids: fixed-length int32 sequences)
  → artifacts/tokenizer/       (trained from data/deduped/)
  → artifacts/checkpoints/{1b,7b}/step_NNNNNNN/
```

Every directory that a stage writes to gets a `.done` sentinel file when the stage completes. Stages check `utils.is_done()` at startup and skip if already finished — delete the `.done` file to re-run a stage.

### Shard format

All intermediate data is stored as HuggingFace Arrow datasets saved to disk. Text stages have a `text` column; packed sequences have an `input_ids` column (list of int32). Shards are named `shard_NNNNN/` and enumerated with `rglob("shard_*")`.

### Module layout

```
src/vi_llm/
  dataprep/
    loader.py      # stage 1 — download from HF Hub via pyarrow + HfFileSystem
    cleaner.py     # stage 2 — TextCleaner (filters + transforms), parallel workers
    dedup.py       # stage 3 — MinHash LSH, parallel hashvalue computation
    packer.py      # stage 5 — tokenise + pack into fixed-length CLM sequences
    converters.py  # registry: dataset-id → row-to-text function
    utils.py       # is_done / mark_done sentinel helpers
  tokenizer/
    train.py       # stage 4 — BPE tokenizer training (HuggingFace tokenizers lib)
  training/
    trainer.py     # stage 6 — Accelerate training loop, PackedArrowDataset
    callbacks.py   # save_checkpoint / push_checkpoint_to_hub
scripts/           # thin wrappers that call the library functions above
configs/
  dataprep.yaml    # stages 1–3 config (datasets list, filter thresholds, num_workers)
  tokenizer.yaml   # stage 4–5 config (vocab size, BPE settings, seq_length)
  training/1b.yaml # model arch, training hyperparams, checkpoint/logging config
  training/7b.yaml
```

### Adding a dataset

Two options (no code changes needed if using `text_columns`):

1. **Simple columns** — add an entry to `configs/dataprep.yaml` under `download.datasets` with `text_columns: [col1, col2]`. Columns are joined with `\n\n`.
2. **Custom converter** — register a function in `src/vi_llm/dataprep/converters.py` with the `@register("org/dataset-id")` decorator for datasets requiring non-trivial row→text transformation (title prepending, multi-field joining, etc.). Converters take priority over `text_columns`.

### Parallel processing

Stages 2, 3, and 5 use `ProcessPoolExecutor` with `mp_context="fork"`. Workers are initialized once with a shared state object (cleaner or tokenizer). `num_workers` in `configs/dataprep.yaml` controls parallelism; set to 1 to use the main process.

### Training details

- Model: `LlamaForCausalLM` built from scratch (no pretrained weights).
- dtype is resolved per device in `trainer.py:resolve_dtype()`, not via Accelerate's `mixed_precision` flag. `accelerate_config.yaml` sets `mixed_precision: no`.
- Checkpoints save full Accelerate state via `accelerator.save_state()`; training auto-resumes from the highest-numbered `step_NNNNNNN/` directory.
- `torch.compile: true` in training YAML enables `torch.compile`. Disable for FSDP multi-GPU runs.
- Multi-GPU FSDP: edit `accelerate_config_fsdp.yaml` to set `num_processes`; NCCL env vars are baked into the `train-{1b,7b}-fsdp` Makefile targets for the local network interface (`enp5s0`).

### Known quirks

- **FastText + NumPy ≥ 2.0**: `cleaner.py:TextCleaner._load_fasttext()` patches `fasttext.FastText.np` with a shim to avoid a `copy=False` breakage. Do not remove this patch.
- **Large string columns**: the loader reads raw parquet via `pyarrow + HfFileSystem` instead of `datasets.load_dataset` to avoid a `large_string → string` cast error in the HF streaming path. The `use_fallback` path calls `datasets.load_dataset` only when parquet files cannot be located.
- **VnCoreNLP word segmentation**: disabled by default (`word_segment.enabled: false`). When enabled, `py_vncorenlp` downloads model files to `artifacts/vncorenlp/` on first run.
