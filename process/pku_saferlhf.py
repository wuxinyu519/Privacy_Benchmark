#!/usr/bin/env python3
"""
Process PKU-Alignment/PKU-SafeRLHF dataset into JSONL for gpt_tagger.py

Filters prompts where Endangering National Security = True
(in either response_0_harm_category or response_1_harm_category)

Usage:
    python pku_saferlhf.py <input_path> <output_dir> [--num_samples N]

Example:
    python pku_saferlhf.py ../datasets/pku_saferlhf ./processed_data --num_samples 10

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
        print("Downloading from HuggingFace: PKU-Alignment/PKU-SafeRLHF ...")
        input_path.mkdir(parents=True, exist_ok=True)
        try:
            import urllib.request
            base = "https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF/resolve/refs%2Fconvert%2Fparquet/default"
            for split in ['train', 'test']:
                url = f"{base}/{split}/0000.parquet"
                out = input_path / f"{split}.parquet"
                print(f"  Downloading {split}...")
                urllib.request.urlretrieve(url, str(out))
            print("Download complete.")
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


def is_national_security(harm_category):
    """Check if Endangering National Security is True in a harm_category dict."""
    if isinstance(harm_category, dict):
        return harm_category.get("Endangering National Security", False)
    if isinstance(harm_category, str):
        try:
            d = json.loads(harm_category)
            return d.get("Endangering National Security", False)
        except:
            return False
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to data dir or parquet file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Randomly sample N items (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(args.input_path)
    print(f"Loaded {len(df)} rows total")

    # Filter: Endangering National Security = True in either response
    # Combine prompt with the corresponding harmful response
    results = []
    seen = set()

    for _, row in df.iterrows():
        prompt_text = row.get('prompt', '')
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            continue

        r0_ns = is_national_security(row.get('response_0_harm_category'))
        r1_ns = is_national_security(row.get('response_1_harm_category'))

        if not r0_ns and not r1_ns:
            continue

        # Pick the response that has Endangering National Security = True
        # If both, pick response_0
        if r0_ns:
            response = row.get('response_0', '')
        else:
            response = row.get('response_1', '')

        if not isinstance(response, str) or not response.strip():
            continue

        # Deduplicate by prompt
        key = prompt_text.strip()
        if key in seen:
            continue
        seen.add(key)

        combined = f"{prompt_text.strip()}\n\nResponse:\n{response.strip()}"
        results.append({"prompt": combined})

    print(f"Endangering National Security prompts: {len(results)}")

    if args.num_samples and args.num_samples < len(results):
        random.seed(42)
        results = random.sample(results, args.num_samples)
        print(f"Randomly sampled {args.num_samples}")

    output_path = os.path.join(args.output_dir, "pku_national_security.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Saved to: {output_path} ({len(results)} samples)")


if __name__ == '__main__':
    main()