Implement a missing vi-llm-pretrain pipeline stage module.

The argument is the module name, e.g. `/implement-stage packer`.

## Steps

1. **Read the test file** `tests/test_<module>.py` — it defines the exact public API (function names, class names, expected behaviours). Never implement without reading this first.

2. **Read the script entry point** `scripts/0N_*.py` that imports from the module — it shows the entry-point function signature and CLI args.

3. **Check memory** for known issues and pipeline conventions (sentinel pattern, Arrow shard format, parallel workers).

4. **Implement** `src/vi_llm/dataprep/<module>.py` following these conventions:
   - Start with `is_done` check, end with `mark_done`
   - Read shards: `hf_datasets.load_from_disk(str(shard_dir))["text"]`
   - Write shards: `hf_datasets.Dataset.from_dict({"text": texts}).save_to_disk(str(shard_path))`, directory name `shard_{idx:05d}`
   - Progress: `print(f"  [{i}/{total}] ...", end="\r")` then `print()` after loop

5. **Run tests**: `pytest tests/test_<module>.py -v` — all must pass before proceeding.

6. **Update CLAUDE.md** — remove any "not yet implemented" note and add any non-obvious behaviour.

## Pipeline conventions

- Intermediate format: HuggingFace Arrow shards in `shard_XXXXX/` directories, single `text` column (or `input_ids` for packer output)
- Sentinel: `vi_llm.dataprep.utils.is_done(path)` / `mark_done(path)` writes `.done` file
- Package: `vi_llm.dataprep.<module>` — import path matches `src/vi_llm/dataprep/<module>.py`
- Data flow: `data/raw → data/cleaned → data/deduped → data/packed → artifacts/`
