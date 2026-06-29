"""Inference — load a checkpoint and generate text."""

from __future__ import annotations

from pathlib import Path
import torch
import yaml
from transformers import PreTrainedTokenizerFast

from vi_llm.training.trainer import build_model


def _find_latest_checkpoint(output_dir: str) -> Path | None:
    p = Path(output_dir)
    if not p.exists():
        return None
    steps = [
        d for d in p.iterdir()
        if d.is_dir() and d.name.startswith("step_") and any(d.iterdir())
    ]
    if not steps:
        return None
    return max(steps, key=lambda d: int(d.name.replace("step_", "")))


def load_for_inference(config_path: str, checkpoint_dir: str | None = None):
    """Return (model, tokenizer) ready for inference.

    Rebuilds the model from the training YAML (same as trainer.py) and loads
    weights from model.safetensors / pytorch_model.bin in the checkpoint dir.
    Avoids from_pretrained so the custom 1B architecture and vocab size are used.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tokenizer_dir = cfg["data"]["tokenizer_dir"]
    ckpt_output_dir = cfg["checkpoint"]["output_dir"]
    dtype_str = cfg["training"].get("dtype", "bf16")
    mixed_precision = "bf16" if dtype_str == "bf16" else "fp16" if dtype_str == "fp16" else "bf16"

    if checkpoint_dir is None:
        latest = _find_latest_checkpoint(ckpt_output_dir)
        if latest is None:
            raise FileNotFoundError(
                f"No checkpoints found in {ckpt_output_dir}. "
                "Run training first or pass --checkpoint explicitly."
            )
        checkpoint_dir = str(latest)
        print(f"Auto-selected checkpoint: {checkpoint_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

    model_cfg = {**cfg["model"], "vocab_size": tokenizer.vocab_size}
    model = build_model(model_cfg, mixed_precision)

    # Load weights — prefer safetensors, fall back to pytorch_model.bin
    ckpt_path = Path(checkpoint_dir)
    safetensors_file = ckpt_path / "model.safetensors"
    bin_file = ckpt_path / "pytorch_model.bin"

    if safetensors_file.exists():
        from safetensors.torch import load_model as st_load
        st_load(model, str(safetensors_file))
    elif bin_file.exists():
        state_dict = torch.load(str(bin_file), map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
    else:
        raise FileNotFoundError(
            f"No model weights found in {checkpoint_dir}. "
            "Expected model.safetensors or pytorch_model.bin."
        )

    model = model.to(device)
    model.eval()

    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    do_sample: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
