Add a new HuggingFace dataset to the vi-llm-pretrain pipeline.

The argument is the dataset ID, e.g. `/add-dataset uonlp/CulturaX`.

## Steps

1. **Inspect the dataset schema** to find the parquet layout and column names:

```python
from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq, pyarrow.fs as pa_fs

fs = HfFileSystem()
files = sorted(fs.glob("datasets/<org>/<name>/**/*.parquet"))
print(files[:5])

pf = pq.ParquetFile(files[0], filesystem=pa_fs.PyFileSystem(pa_fs.FSSpecHandler(fs)))
print(pf.schema_arrow)
batch = next(pf.iter_batches(batch_size=2))
for col in pf.schema_arrow.names:
    print(f"  {col!r}: {str(batch.column(col)[0].as_py())[:120]!r}")
```

2. **Identify** the correct `name` (config subset) and `split` from the file paths. Common issues:
   - Split may not be `train` — check actual file names
   - Config name required if multiple subsets exist

3. **Register a converter** in `src/vi_llm/dataprep/converters.py`:

```python
@register("org/dataset-id")
def _(row: dict) -> str | None:
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    if not text:
        return None
    return f"# {title}\n\n{text}" if title else text
```

Return `None` to skip a row. For simple single-column datasets, `text_columns` in the YAML is enough — no converter needed.

4. **Add entry to `configs/datasets.yaml`**:

```yaml
- id: org/dataset-id
  name: subset_name     # omit if no subsets
  split: train
  weight: 1.0           # relative training-time sampling weight
  max_docs: null        # set an int for smoke tests
```

5. **Verify** the loader finds the files and produces output:

```bash
# Temporarily set max_docs: 100 in datasets.yaml, then:
make download
```

Check `data/raw/<org>__<name>/shard_00000/` exists and contains documents.
