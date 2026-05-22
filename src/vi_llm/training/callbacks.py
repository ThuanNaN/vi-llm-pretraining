"""Training callbacks: checkpointing and HF Hub push."""

from __future__ import annotations

import os
from pathlib import Path


def save_checkpoint(accelerator, model, output_dir: str, step: int) -> None:
    path = Path(output_dir) / f"step_{step:07d}"
    path.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(str(path))


def push_checkpoint_to_hub(checkpoint_dir: str, repo_id: str) -> None:
    from huggingface_hub import HfApi

    if not repo_id:
        repo_id = os.environ.get("HF_CHECKPOINT_REPO")
    if not repo_id:
        raise ValueError("Set checkpoint.hf_repo in training YAML or HF_CHECKPOINT_REPO env var.")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Checkpoint pushed to https://huggingface.co/{repo_id}")
