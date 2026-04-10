#!/usr/bin/env python3
"""
Process Rohit-D/synthetic-confidential-information-injected-business-excerpts into JSONL

Excerpt field -> prompt

Usage:
    python confidential_biz.py <input_path> <output_dir> [--num_samples N]

Example:
    python confidential_biz.py ../datasets/confidential_biz ./processed_data --num_samples 10

Auto-downloads from HuggingFace if local path doesn't exist.
"""

import json
import random
import argparse
import os
from pathlib import Path


def load_data(input_path):
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas is required. Install with: pip install pandas pyarrow")
        exit(1)

    input_path = Path(input_path)

    # Auto-download if needed
    needs_download = False
    if not input_path.exists():
        needs_download = True
    elif input_path.is_dir() and not list(input_path.rglob('*.csv')) and not list(input_path.rglob('*.parquet')):
        needs_download = True

    if needs_download:
        print(f"No data found at: {input_path}")
        print("Downloading from HuggingFace ...")
        input_path.mkdir(parents=True, exist_ok=True)
        try:
            import urllib.request
            # Try auto-converted parquet first
            url = "https://huggingface.co/datasets/Rohit-D/synthetic-confidential-information-injected-business-excerpts/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
            parquet_path = input_path / "train.parquet"
            urllib.request.urlretrieve(url, str(parquet_path))
            print(f"Saved to {parquet_path}")
        except Exception as e:
            print(f"Error downloading: {e}")
            exit(1)

    # Load
    parquet_files = list(input_path.rglob('*.parquet'))
    csv_files = list(input_path.rglob('*.csv'))

    if input_path.is_file():
        if input_path.suffix == '.parquet':
            return pd.read_parquet(input_path)
        elif input_path.suffix == '.csv':
            return pd.read_csv(input_path)

    if parquet_files:
        return pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    elif csv_files:
        return pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    else:
        print(f"No parquet or csv files found in {input_path}")
        exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to data dir or file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Randomly sample N items (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(args.input_path)
    print(f"Loaded {len(df)} rows")

    results = []
    for _, row in df.iterrows():
        text = row.get('Excerpt', '')
        if isinstance(text, str) and text.strip():
            results.append({"prompt": text.strip()})

    print(f"Valid excerpts: {len(results)}")

    if args.num_samples and args.num_samples < len(results):
        random.seed(42)
        results = random.sample(results, args.num_samples)
        print(f"Randomly sampled {args.num_samples}")

    output_path = os.path.join(args.output_dir, "confidential_biz.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Saved to: {output_path} ({len(results)} samples)")


if __name__ == '__main__':
    main()