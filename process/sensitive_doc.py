#!/usr/bin/env python3
"""
Process mouhamet/sensitive_document_classification into JSONL for gpt_tagger.py

Original data is a zip of .txt files. Each .txt = one full document = one prompt.
Filename convention: [model]_[language]_[label]_[guideline]_[random-number].txt

Usage:
    python sensitive_doc.py <input_path> <output_dir> [--num_samples N]

Example:
    python sensitive_doc.py ../datasets/sensitive_document_classification ./processed_data --num_samples 10

Input can be:
    - Path to extracted directory containing .txt files
    - Path to dataset.zip
    - If path doesn't exist, auto-downloads from HuggingFace
"""

import json
import random
import argparse
import os
import zipfile
from pathlib import Path


def ensure_data(input_path):
    """Ensure data is available. Download and extract if needed."""
    input_path = Path(input_path)

    # Check if .txt files already exist (extracted)
    if input_path.is_dir():
        txt_files = list(input_path.rglob('*.txt'))
        if txt_files:
            return input_path

    # Check if it's a zip file
    if input_path.is_file() and input_path.suffix == '.zip':
        extract_dir = input_path.parent / "sensitive_doc_extracted"
        print(f"Extracting {input_path} ...")
        with zipfile.ZipFile(input_path, 'r') as z:
            z.extractall(extract_dir)
        return extract_dir

    # Auto-download
    print(f"No data found at: {input_path}")
    print("Downloading from HuggingFace: mouhamet/sensitive_document_classification ...")
    input_path.mkdir(parents=True, exist_ok=True)

    zip_path = input_path / "dataset.zip"
    try:
        import urllib.request
        url = "https://huggingface.co/datasets/mouhamet/sensitive_document_classification/resolve/main/dataset.zip"
        urllib.request.urlretrieve(url, str(zip_path))
        print(f"Downloaded to {zip_path}")
    except Exception as e:
        print(f"Error downloading: {e}")
        exit(1)

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(input_path)
    print("Done.")

    return input_path


def load_documents(data_dir):
    """Read each .txt file as one full document."""
    data_dir = Path(data_dir)
    txt_files = sorted(data_dir.rglob('*.txt'))
    print(f"Found {len(txt_files)} .txt files")

    documents = []
    for fp in txt_files:
        text = fp.read_text(encoding='utf-8', errors='ignore').strip()
        if text:
            documents.append({"prompt": text})

    return documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to data dir, zip file, or download target")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Randomly sample N items (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data_dir = ensure_data(args.input_path)
    results = load_documents(data_dir)
    print(f"Total documents: {len(results)}")

    if args.num_samples and args.num_samples < len(results):
        random.seed(42)
        results = random.sample(results, args.num_samples)
        print(f"Randomly sampled {args.num_samples}")

    output_path = os.path.join(args.output_dir, "sensitive_doc.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Saved to: {output_path} ({len(results)} samples)")


if __name__ == '__main__':
    main()