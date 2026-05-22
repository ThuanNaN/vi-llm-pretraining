.PHONY: install install-flash install-cuda download clean-data dedup tokenizer \
        tokenize train-1b train-7b train-1b-fsdp train-7b-fsdp \
        pipeline push-tokenizer push-checkpoint test

PYTHON  := python
SCRIPTS := scripts
CONFIGS := configs

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	uv pip install -e .

install-flash:
	uv pip install -e ".[flash]"      # CUDA only — Flash Attention 2

install-cuda:
	uv pip install -e ".[cuda]"       # Linux/CUDA only — adds DeepSpeed

# ── Data pipeline ─────────────────────────────────────────────────────────────

download:
	$(PYTHON) $(SCRIPTS)/01_download.py --config $(CONFIGS)/datasets.yaml

clean-data:
	$(PYTHON) $(SCRIPTS)/02_clean.py --config $(CONFIGS)/cleaning.yaml

dedup:
	$(PYTHON) $(SCRIPTS)/03_dedup.py

tokenizer:
	$(PYTHON) $(SCRIPTS)/04_train_tokenizer.py --config $(CONFIGS)/tokenizer.yaml

tokenize:
	$(PYTHON) $(SCRIPTS)/05_tokenize.py --config $(CONFIGS)/tokenizer.yaml

pipeline: download clean-data dedup tokenizer tokenize

# ── Training — single device (CUDA or MPS, auto-detected) ────────────────────
# Uses accelerate_config.yaml — works on both Apple Silicon and CUDA.
# dtype and attn_implementation are resolved automatically by trainer.py.

train-1b:
	accelerate launch --config_file accelerate_config.yaml \
		$(SCRIPTS)/06_train.py --config $(CONFIGS)/training/1b.yaml

train-7b:
	accelerate launch --config_file accelerate_config.yaml \
		$(SCRIPTS)/06_train.py --config $(CONFIGS)/training/7b.yaml

# ── Training — multi-GPU FSDP (CUDA only) ────────────────────────────────────
# Edit accelerate_config_fsdp.yaml to set num_processes = GPU count.

train-1b-fsdp:
	accelerate launch --config_file accelerate_config_fsdp.yaml \
		$(SCRIPTS)/06_train.py --config $(CONFIGS)/training/1b.yaml

train-7b-fsdp:
	accelerate launch --config_file accelerate_config_fsdp.yaml \
		$(SCRIPTS)/06_train.py --config $(CONFIGS)/training/7b.yaml

# ── HF Hub ────────────────────────────────────────────────────────────────────

push-tokenizer:
	$(PYTHON) -c "from vi_llm.tokenizer.train import push_to_hub; push_to_hub('artifacts/tokenizer')"

push-checkpoint:
	$(PYTHON) -c "from vi_llm.training.callbacks import push_checkpoint_to_hub; push_checkpoint_to_hub()"

# ── Tests ──────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v
