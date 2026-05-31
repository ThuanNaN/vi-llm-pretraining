# vi-llm-pretrain

End-to-end pre-training pipeline for a Vietnamese causal language model (1–7B parameters).

| Stage | Script | Description |
|---|---|---|
| 1 | `scripts/01_download.py` | Download datasets from HuggingFace Hub as Arrow shards |
| 2 | `scripts/02_clean.py` | Quality filtering + language ID (FastText) |
| 3 | `scripts/03_dedup.py` | MinHash LSH near-duplicate removal |
| 4 | `scripts/04_train_tokenizer.py` | Train BPE tokenizer (65k vocab) |
| 5 | `scripts/05_tokenize.py` | Tokenise + pack into fixed-length sequences |
| 6 | `scripts/06_train.py` | Pre-training via Accelerate (single GPU or FSDP) |

## Setup

```bash
pip install uv
uv pip install -e .

# Linux/CUDA only — adds DeepSpeed
uv pip install -e ".[cuda]"
```

Copy `.env.example` to `.env` and set `HF_TOKEN` to push to the Hub.

## Run the full data pipeline

```bash
make pipeline          # stages 1–5
```

For a quick smoke test, set `max_docs: 1000` on a dataset entry in `configs/dataprep.yaml`.

## Training

```bash
# Single GPU / Apple Silicon MPS
make train-1b          # LLaMA-style 1B
make train-7b          # LLaMA-style 7B

# Multi-GPU FSDP (CUDA only)
# Edit accelerate_config_fsdp.yaml to set num_processes = GPU count
make train-1b-fsdp
make train-7b-fsdp
```

Training automatically resumes from the latest checkpoint in `artifacts/checkpoints/`.

## Adding a new dataset

Edit `configs/dataprep.yaml` under `download.datasets` — no code changes required for simple cases:

```yaml
datasets:
  - id: your-org/your-dataset
    split: train
    text_columns: [text]   # columns whose text will be joined with \n\n
```

For datasets needing custom row→text logic, register a converter in `src/vi_llm/dataprep/converters.py` with the `@register("your-org/your-dataset")` decorator. Converters take priority over `text_columns`.

## Speed-up options

| Option | How to enable |
|---|---|
| Flash Attention 2 | `attn_implementation: flash_attention_2` in training YAML |
| `torch.compile` | `compile: true` in training YAML (disable for FSDP runs) |
| DeepSpeed ZeRO-3 | Replace `accelerate_config.yaml` with a DeepSpeed plugin config |
| Gradient checkpointing | `gradient_checkpointing: true` (default on) |

## HF Hub push

```bash
make push-tokenizer    # push artifacts/tokenizer/ to HF Hub
make push-checkpoint   # push latest checkpoint to HF Hub
```

Set `hf_repo` in the respective YAML config or export `HF_TOKENIZER_REPO` / `HF_CHECKPOINT_REPO`.

## Tests

```bash
make test
```
