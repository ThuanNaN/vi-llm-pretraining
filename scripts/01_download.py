#!/usr/bin/env python
"""Stage 1 — Download datasets from HuggingFace Hub."""

import argparse

from vi_llm.dataprep.loader import download_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataprep.yaml")
    args = parser.parse_args()
    download_all(args.config)
