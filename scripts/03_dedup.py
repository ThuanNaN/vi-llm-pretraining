#!/usr/bin/env python
"""Stage 3 — MinHash LSH near-duplicate removal."""

import argparse

from vi_llm.dataprep.dedup import dedup_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/cleaned")
    parser.add_argument("--output-dir", default="data/deduped")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--num-perm", type=int, default=128)
    args = parser.parse_args()
    dedup_dataset(args.input_dir, args.output_dir, args.threshold, args.num_perm)
