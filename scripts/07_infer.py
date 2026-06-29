#!/usr/bin/env python
"""Stage 7 — Text generation from a trained checkpoint.

Usage:
  python scripts/07_infer.py --config configs/training/1b-smoke.yaml \
      --prompt "Điện thoại này"

  # Explicit checkpoint:
  python scripts/07_infer.py --config configs/training/1b-smoke.yaml \
      --checkpoint artifacts/checkpoints/1b-smoke/step_0000100 \
      --prompt "Điện thoại này" --max-new-tokens 300
"""

import argparse

from vi_llm.training.infer import generate, load_for_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Training YAML config")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint directory (default: latest in config's output_dir)",
    )
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument(
        "--greedy", action="store_true", help="Greedy decoding (disables sampling)"
    )
    args = parser.parse_args()

    model, tokenizer = load_for_inference(args.config, args.checkpoint)

    output = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.greedy,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    print(f"\nPrompt : {args.prompt}")
    print(f"Output : {output}")
