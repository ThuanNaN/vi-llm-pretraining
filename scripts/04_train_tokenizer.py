#!/usr/bin/env python
"""Stage 4 — Train BPE tokenizer on the deduplicated corpus."""

import argparse

from vi_llm.tokenizer.train import train_tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tokenizer.yaml")
    args = parser.parse_args()
    train_tokenizer(args.config)
