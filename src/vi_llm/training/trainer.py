"""Accelerate-based pre-training loop.

Runs on CUDA (single or multi-GPU via FSDP) and Apple Silicon MPS
with no manual config switching — device, dtype, and attention
implementation are resolved automatically at startup.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, IterableDataset
from transformers import LlamaConfig, LlamaForCausalLM

import wandb

from .callbacks import push_checkpoint_to_hub, save_checkpoint


# ── Device / dtype helpers ────────────────────────────────────────────────────

def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype_cfg: str, device: str) -> str:
    """Translate config dtype (possibly 'auto') to a concrete Accelerate value.

    Accelerate accepts: 'no' (fp32), 'fp16', 'bf16'.
    """
    if dtype_cfg != "auto":
        # fp32 is spelled 'no' in Accelerate's mixed_precision API
        return "no" if dtype_cfg == "fp32" else dtype_cfg

    if device == "cuda":
        # bf16 is available on Ampere+ (A100, RTX 3xxx+); safe default for modern GPUs
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if device == "mps":
        # MPS does not support bf16 reliably; fp32 is the safe choice
        return "no"
    return "no"  # CPU


def resolve_attn_impl(impl_cfg: str, device: str) -> str:
    """Translate 'auto' attention implementation to a concrete value."""
    if impl_cfg != "auto":
        return impl_cfg

    if device == "cuda":
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except ImportError:
            return "sdpa"
    # MPS and CPU both support PyTorch's scaled_dot_product_attention
    return "sdpa"


# ── Dataset ───────────────────────────────────────────────────────────────────

class PackedArrowDataset(IterableDataset):
    """Streams packed sequences from Arrow shards on disk."""

    def __init__(self, packed_dir: str):
        import datasets as hf_datasets
        self._shard_dirs = sorted(Path(packed_dir).rglob("shard_*"))
        self._hf_datasets = hf_datasets

    def __iter__(self):
        for shard_dir in self._shard_dirs:
            ds = self._hf_datasets.load_from_disk(str(shard_dir))
            for row in ds:
                ids = torch.tensor(row["input_ids"], dtype=torch.long)
                yield {"input_ids": ids, "labels": ids.clone()}


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(model_cfg: dict, device: str) -> LlamaForCausalLM:
    attn_impl = resolve_attn_impl(
        model_cfg.get("attn_implementation", "auto"), device
    )
    config = LlamaConfig(
        hidden_size=model_cfg["hidden_size"],
        num_hidden_layers=model_cfg["num_hidden_layers"],
        num_attention_heads=model_cfg["num_attention_heads"],
        num_key_value_heads=model_cfg.get("num_key_value_heads", model_cfg["num_attention_heads"]),
        intermediate_size=model_cfg["intermediate_size"],
        max_position_embeddings=model_cfg.get("max_position_embeddings", 2048),
        rms_norm_eps=model_cfg.get("rms_norm_eps", 1e-5),
        rope_theta=model_cfg.get("rope_theta", 10000.0),
        attn_implementation=attn_impl,
    )
    return LlamaForCausalLM(config)


# ── Scheduler ─────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=0)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


# ── Main training entry point ─────────────────────────────────────────────────

def train(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    ckpt_cfg = cfg["checkpoint"]
    log_cfg = cfg["logging"]

    device = detect_device()
    mixed_precision = resolve_dtype(train_cfg.get("dtype", "auto"), device)

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
    )

    if accelerator.is_main_process:
        print(f"[device={device}  mixed_precision={mixed_precision}]")
        wandb.init(project=log_cfg.get("wandb_project", "vi-llm-pretrain"))

    model = build_model(model_cfg, device)

    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.1),
    )

    total_steps = train_cfg["num_train_steps"]
    warmup_steps = train_cfg.get("warmup_steps", 2000)
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    dataset = PackedArrowDataset(data_cfg["packed_dir"])
    dataloader = DataLoader(dataset, batch_size=train_cfg["per_device_batch_size"])

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # torch.compile has limited MPS support — skip on non-CUDA
    if cfg.get("compile", False) and device == "cuda":
        model = torch.compile(model)

    output_dir = ckpt_cfg["output_dir"]
    resume_step = _find_resume_step(output_dir)
    if resume_step > 0 and accelerator.is_main_process:
        accelerator.load_state(str(Path(output_dir) / f"step_{resume_step:07d}"))

    log_every = log_cfg.get("log_every_steps", 10)
    save_every = ckpt_cfg.get("save_every_steps", 1000)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    hf_repo = ckpt_cfg.get("hf_repo")

    global_step = resume_step
    for batch in dataloader:
        if global_step >= total_steps:
            break

        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1

            if global_step % log_every == 0 and accelerator.is_main_process:
                lr = scheduler.get_last_lr()[0]
                wandb.log({"loss": loss.item(), "lr": lr, "step": global_step})

            if global_step % save_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_checkpoint(accelerator, model, output_dir, global_step)
                    if hf_repo:
                        push_checkpoint_to_hub(
                            str(Path(output_dir) / f"step_{global_step:07d}"), hf_repo
                        )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(accelerator, model, output_dir, global_step)
        if hf_repo:
            push_checkpoint_to_hub(
                str(Path(output_dir) / f"step_{global_step:07d}"), hf_repo
            )
        wandb.finish()


def _find_resume_step(output_dir: str) -> int:
    p = Path(output_dir)
    if not p.exists():
        return 0
    steps = [
        int(d.name.replace("step_", ""))
        for d in p.iterdir()
        if d.is_dir() and d.name.startswith("step_")
    ]
    return max(steps, default=0)
