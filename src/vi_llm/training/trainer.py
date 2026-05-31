"""Accelerate-based pre-training loop. Requires CUDA + Flash Attention 2."""

from __future__ import annotations

import os
from pathlib import Path

import bisect
import torch
import yaml
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaForCausalLM
from tqdm import tqdm

import wandb

from .callbacks import push_checkpoint_to_hub, save_checkpoint


# ── dtype helper ──────────────────────────────────────────────────────────────

def resolve_dtype(dtype_cfg: str) -> str:
    """Translate config dtype (possibly 'auto') to a concrete Accelerate value."""
    if dtype_cfg != "auto":
        return "no" if dtype_cfg == "fp32" else dtype_cfg
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


# ── Dataset ───────────────────────────────────────────────────────────────────

class PackedArrowDataset(torch.utils.data.Dataset):
    """Packed sequences from Arrow shards on disk.

    Using a map-style dataset allows Accelerate to use a distributed sampler and avoids
    the rank-0-only broadcast path that can deadlock under FSDP / multi-process dataloader.
    """

    def __init__(self, packed_dir: str):
        import datasets as hf_datasets

        self._shard_dirs = sorted(Path(packed_dir).rglob("shard_*"))
        self._shards = [hf_datasets.load_from_disk(str(shard_dir)) for shard_dir in self._shard_dirs]
        self._lengths = [len(shard) for shard in self._shards]
        self._cumulative_lengths = [0]
        for length in self._lengths:
            self._cumulative_lengths.append(self._cumulative_lengths[-1] + length)

    def __len__(self) -> int:
        return self._cumulative_lengths[-1]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError("index out of range")

        shard_idx = bisect.bisect_right(self._cumulative_lengths, idx) - 1
        local_idx = idx - self._cumulative_lengths[shard_idx]
        row = self._shards[shard_idx][local_idx]
        ids = torch.tensor(row["input_ids"], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(model_cfg: dict, mixed_precision: str) -> LlamaForCausalLM:
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    torch_dtype = dtype_map.get(mixed_precision, torch.bfloat16)

    config = LlamaConfig(
        vocab_size=model_cfg.get("vocab_size", 32000),
        hidden_size=model_cfg["hidden_size"],
        num_hidden_layers=model_cfg["num_hidden_layers"],
        num_attention_heads=model_cfg["num_attention_heads"],
        num_key_value_heads=model_cfg.get("num_key_value_heads", model_cfg["num_attention_heads"]),
        intermediate_size=model_cfg["intermediate_size"],
        max_position_embeddings=model_cfg.get("max_position_embeddings", 2048),
        rms_norm_eps=model_cfg.get("rms_norm_eps", 1e-5),
        rope_parameters={"rope_type": "default", "rope_theta": model_cfg.get("rope_theta", 10000.0)},
        attention_dropout=model_cfg.get("attention_dropout", 0.0),
        attention_bias=model_cfg.get("attention_bias", False),
        mlp_bias=model_cfg.get("mlp_bias", False),
        tie_word_embeddings=model_cfg.get("tie_word_embeddings", False),
        bos_token_id=model_cfg.get("bos_token_id", 1),
        eos_token_id=model_cfg.get("eos_token_id", 2),
        pad_token_id=model_cfg.get("pad_token_id", None),
        attn_implementation="flash_attention_2",
        torch_dtype=torch_dtype,
    )

    # Build directly under target dtype so weights are initialized in bf16/fp16.
    prev_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch_dtype)
        model = LlamaForCausalLM(config)
    finally:
        torch.set_default_dtype(prev_dtype)

    model = model.to(dtype=torch_dtype)

    # For training with gradient checkpointing / FSDP.
    model.config.use_cache = False

    return model

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

    mixed_precision = resolve_dtype(train_cfg.get("dtype", "auto"))

    accelerator = Accelerator(
        mixed_precision="no",  # dtype handled in code via build_model; avoids FSDP param upcasting
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
    )

    if accelerator.is_main_process:
        print(f"[mixed_precision={mixed_precision}]")
        wandb.init(
            project=log_cfg.get("wandb_project", "vi-llm-pretrain"),
            mode=os.environ.get("WANDB_MODE", "offline"),
        )

    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(data_cfg["tokenizer_dir"])
    model_cfg = {**model_cfg, "vocab_size": tokenizer.vocab_size}

    model = build_model(model_cfg, mixed_precision)

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
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["per_device_batch_size"],
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=True,
        drop_last=True,
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    if cfg.get("compile", False):
        model = torch.compile(model)

    output_dir = ckpt_cfg["output_dir"]
    resume_step = _find_resume_step(output_dir)
    if resume_step > 0:
        accelerator.load_state(str(Path(output_dir) / f"step_{resume_step:07d}"))

    log_every = log_cfg.get("log_every_steps", 10)
    save_every = ckpt_cfg.get("save_every_steps", 1000)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    hf_repo = ckpt_cfg.get("hf_repo")

    def cycling_loader(dl):
        while True:
            yield from dl

    global_step = resume_step
    pbar = tqdm(
        total=total_steps,
        initial=resume_step,
        desc="Training",
        unit="step",
        disable=not accelerator.is_main_process,
    )
    for batch in cycling_loader(dataloader):
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
            lr = scheduler.get_last_lr()[0]

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

            if global_step % log_every == 0 and accelerator.is_main_process:
                wandb.log({"loss": loss.item(), "lr": lr, "step": global_step})

            if global_step % save_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_checkpoint(accelerator, model, output_dir, global_step)
                    if hf_repo:
                        push_checkpoint_to_hub(
                            str(Path(output_dir) / f"step_{global_step:07d}"), hf_repo
                        )

    pbar.close()
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
        if d.is_dir() and d.name.startswith("step_") and any(d.iterdir())
    ]
    return max(steps, default=0)
