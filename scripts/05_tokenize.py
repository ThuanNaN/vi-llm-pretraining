#!/usr/bin/env python
"""Stage 5 — Tokenise corpus and pack into fixed-length CLM sequences."""

import argparse

import yaml

from vi_llm.data.packer import pack_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tokenizer.yaml")
    parser.add_argument("--input-dir", default="data/deduped")
    parser.add_argument("--output-dir", default="data/packed")
    parser.add_argument("--seq-length", type=int, default=2048)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pack_dataset(
        tokenizer_dir=cfg["output_dir"],
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        seq_length=args.seq_length,
    )
