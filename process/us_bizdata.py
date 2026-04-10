#!/usr/bin/env python3
"""
Process ttn1410/US_BussinessData into JSONL for gpt_tagger.py

description field -> prompt

Usage:
    python us_bizdata.py <input_path> <output_dir> [--num_samples N]

Example:
    python us_bizdata.py ../datasets/us_bizdata ./processed_data --num_samples 10

Auto-downloads parquet from HuggingFace if local path doesn't exist.
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

    needs_download = False
    if not input_path.exists():
        needs_download = True
    elif input_path.is_dir() and not list(input_path.rglob('*.parquet')):
        needs_download = True

    if needs_download:
        print(f"No data found at: {input_path}")
        print("Downloading from HuggingFace: ttn1410/US_BussinessData ...")
        input_path.mkdir(parents=True, exist_ok=True)
        try:
            import urllib.request
            url = "https://huggingface.co/datasets/ttn1410/US_BussinessData/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
            out = input_path / "train.parquet"
            urllib.request.urlretrieve(url, str(out))
            print(f"Saved to {out}")
        except Exception as e:
            print(f"Error downloading: {e}")
            exit(1)

    if input_path.is_file() and input_path.suffix == '.parquet':
        return pd.read_parquet(input_path)
    elif input_path.is_dir():
        parquet_files = list(input_path.rglob('*.parquet'))
        if not parquet_files:
            print(f"No parquet files found in {input_path}")
            exit(1)
        return pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    else:
        print(f"Invalid input path: {input_path}")
        exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to data dir or parquet file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Randomly sample N items (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(args.input_path)
    print(f"Loaded {len(df)} rows")

    results = []
    for _, row in df.iterrows():
        text = row.get('description', '')
        if isinstance(text, str) and text.strip():
            results.append({"prompt": text.strip()})

    print(f"Valid descriptions: {len(results)}")

    if args.num_samples and args.num_samples < len(results):
        random.seed(42)
        results = random.sample(results, args.num_samples)
        print(f"Randomly sampled {args.num_samples}")

    output_path = os.path.join(args.output_dir, "us_bizdata.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Saved to: {output_path} ({len(results)} samples)")


if __name__ == '__main__':
    main()