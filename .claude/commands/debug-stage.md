Debug a failing vi-llm-pretrain pipeline stage.

The argument is the make target, e.g. `/debug-stage clean-data`.

## Diagnostic checklist

### 1. Check prerequisite stage output
Each stage reads from the previous stage's output directory:
- `download` → `data/raw/`
- `clean-data` → `data/cleaned/`
- `dedup` → `data/deduped/`
- `tokenize` → `data/packed/`

Verify the input directory has shards:
```bash
find data/<input_dir> -type d -name "shard_*" | wc -l
```

### 2. Check sentinel files
If a stage exits silently with no output, it may have been skipped:
```bash
find data/ -name ".done"
```
Delete `.done` in the relevant output directory to force a re-run.

### 3. Inspect a shard's content and schema
```python
import datasets
ds = datasets.load_from_disk("data/<dir>/shard_00000")
print(len(ds), ds.column_names)
print(ds["text"][0][:300])
```

### 4. Check for 0-document output
If a stage produces `Saved 0 documents`:
- The column name in the shard may not match what the stage expects
- For `data/raw`: columns come from the converter or `text_columns` config
- Inspect with the script above

### 5. Common known issues
- **FastText NumPy 2.x**: `ValueError: Unable to avoid copy` → already patched in `cleaner.py`; if it resurfaces, check that `_ft_module._np2_patched` is being set
- **`large_string` cast error**: only in HF `datasets` streaming — loader uses pyarrow directly, so this shouldn't occur in `loader.py`
- **`data/` gitignore**: `.gitignore` must have `/data/` (anchored) not `data/` — otherwise `src/vi_llm/dataprep/` gets ignored
- **VnCoreNLP**: requires Java; `save_dir` must exist before `download_model` is called

### 6. Run the script directly with full traceback
```bash
python scripts/0N_<stage>.py 2>&1 | tail -40
```
`make` sometimes suppresses tracebacks — running the script directly shows the full error.
