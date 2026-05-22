#!/usr/bin/env python
"""Stage 2 — Clean and quality-filter raw text."""

import argparse

from vi_llm.data.cleaner import clean_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cleaning.yaml")
    args = parser.parse_args()
    clean_dataset(args.config)
