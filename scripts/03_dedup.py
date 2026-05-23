#!/usr/bin/env python
"""Stage 3 — MinHash LSH near-duplicate removal."""

import argparse

from vi_llm.dataprep.dedup import dedup_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataprep.yaml")
    args = parser.parse_args()
    dedup_dataset(args.config)
