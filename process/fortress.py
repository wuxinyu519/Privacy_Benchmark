#!/usr/bin/env python3
"""
Process ScaleAI/fortress_public dataset into two JSONL files for gpt_tagger.py
  - fortress_benign.jsonl   (from benign_prompt)
  - fortress_risky.jsonl    (from adversarial_prompt)

Usage:
    python fortress.py <input_path> <output_dir> [--num_samples N]

Example:
    python fortress.py ../datasets/fortress_public ./processed_data --num_samples 10

Input can be:
    - A parquet file (e.g., train.parquet)
    - A directory containing parquet file(s)
"""

import json
import random
import argparse
import os
from pathlib import Path


def load_data(input_path):
    """Load data from parquet file or directory of parquet files."""
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas is required. Install with: pip install pandas pyarrow")
        exit(1)

    input_path = Path(input_path)

    # Auto-download from HuggingFace if local path doesn't exist
    if not input_path.exists():
        print(f"Local path not found: {input_path}")
        print("Downloading from HuggingFace: ScaleAI/fortress_public ...")
        try:
            url = "https://huggingface.co/datasets/ScaleAI/fortress_public/resolve/main/data/train-00000-of-00001.parquet"
            input_path.mkdir(parents=True, exist_ok=True)
            parquet_path = input_path / "train.parquet"

            import urllib.request
            urllib.request.urlretrieve(url, str(parquet_path))
            print(f"Saved to {parquet_path}")
        except Exception as e:
            print(f"Error downloading: {e}")
            exit(1)

    if input_path.is_file() and input_path.suffix == '.parquet':
        df = pd.read_parquet(input_path)
    elif input_path.is_dir():
        parquet_files = list(input_path.rglob('*.parquet'))
        if not parquet_files:
            print(f"No parquet files found in {input_path}")
            exit(1)
        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    else:
        print(f"Invalid input path: {input_path}")
        exit(1)

    return df


def save_jsonl(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to parquet file, directory, or HF dataset name")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Randomly sample N items per split (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(args.input_path)
    print(f"Loaded {len(df)} rows")

    # Build benign list
    benign = []
    for _, row in df.iterrows():
        text = row.get('benign_prompt', '')
        if isinstance(text, str) and text.strip():
            benign.append({"prompt": text.strip()})

    # Build risky list
    risky = []
    for _, row in df.iterrows():
        text = row.get('adversarial_prompt', '')
        if isinstance(text, str) and text.strip():
            risky.append({"prompt": text.strip()})

    print(f"Benign: {len(benign)}, Risky: {len(risky)}")

    # Sample if needed
    if args.num_samples:
        random.seed(42)
        if args.num_samples < len(benign):
            benign = random.sample(benign, args.num_samples)
            print(f"Sampled {args.num_samples} benign")
        if args.num_samples < len(risky):
            risky = random.sample(risky, args.num_samples)
            print(f"Sampled {args.num_samples} risky")

    benign_path = os.path.join(args.output_dir, "fortress_benign.jsonl")
    risky_path = os.path.join(args.output_dir, "fortress_risky.jsonl")

    save_jsonl(benign, benign_path)
    save_jsonl(risky, risky_path)

    print(f"Saved: {benign_path} ({len(benign)} samples)")
    print(f"Saved: {risky_path} ({len(risky)} samples)")


if __name__ == '__main__':
    main()