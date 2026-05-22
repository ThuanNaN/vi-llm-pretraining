# vi-llm-pretrain

End-to-end pre-training pipeline for a Vietnamese causal language model (1–7B parameters).

| Stage | Script | Description |
|---|---|---|
| 1 | `scripts/01_download.py` | Stream datasets from HuggingFace Hub |
| 2 | `scripts/02_clean.py` | Quality filtering + language ID |
| 3 | `scripts/03_dedup.py` | MinHash LSH near-duplicate removal |
| 4 | `scripts/04_train_tokenizer.py` | Train BPE tokenizer (65k vocab) |
| 5 | `scripts/05_tokenize.py` | Tokenise + pack into fixed-length sequences |
| 6 | `scripts/06_train.py` | Pre-training via Accelerate + FSDP |

## Setup

```bash
pip install uv
uv pip install -e .

# Optional: Flash Attention 2 (requires CUDA)
uv pip install -e ".[flash]"
```

Copy `.env.example` to `.env` and set `HF_TOKEN` if you want to push to the Hub.

## Run the full data pipeline

```bash
make pipeline          # stages 1–5
```

For a quick smoke test on 1 000 documents, set `max_docs: 1000` in `configs/datasets.yaml`.

## Training

```bash
# Edit accelerate_config.yaml to set num_processes = number of GPUs
make train-1b          # LLaMA-style 1B
make train-7b          # LLaMA-style 7B
```

## Adding a new dataset

Edit `configs/datasets.yaml` — no code changes required:

```yaml
datasets:
  - id: your-org/your-dataset
    split: train
    text_columns: [text]   # whichever columns hold raw text
    weight: 1.0
```

## Speed-up options

| Option | How to enable |
|---|---|
| Flash Attention 2 | `attn_implementation: flash_attention_2` in training YAML + `make install-flash` |
| `torch.compile` | `compile: true` in training YAML |
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
