#!/usr/bin/env python
"""Stage 6 — Pre-training.

Launch via:
  accelerate launch --config_file accelerate_config.yaml scripts/06_train.py --config configs/training/1b.yaml
"""

import argparse

from vi_llm.training.trainer import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    args = parser.parse_args()
    train(args.config)
