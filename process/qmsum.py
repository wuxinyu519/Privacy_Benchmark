#!/usr/bin/env python3
"""
Process QMSum dataset (Product + Committee) into JSONL format for gpt_tagger.py

Usage:
    python process_qmsum.py <qmsum_data_dir> <output_path> [--num_samples N]

Example:
    python process_qmsum.py ./QMSum/data ./qmsum_product_committee.jsonl --num_samples 10
"""

import json
import random
import argparse
import os
import sys
from pathlib import Path


def extract_context(transcripts, text_spans):
    lines = []
    for span in text_spans:
        start = int(span[0])
        end = int(span[1])
        for i in range(start, min(end + 1, len(transcripts))):
            t = transcripts[i]
            lines.append(f"{t['speaker']}: {t['content']}")
    return "\n".join(lines)


def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    transcripts = data['meeting_transcripts']
    results = []

    for sq in data.get('specific_query_list', []):
        query = sq['query']
        spans = sq.get('relevant_text_span', [])
        context = extract_context(transcripts, spans)

        prompt = (
            f"Based on the following meeting transcript excerpt, answer the query.\n\n"
            f"Query: {query}\n\n"
            f"Context:\n{context}"
        )
        results.append({"prompt": prompt})

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Path to QMSum/data directory")
    parser.add_argument("output_path", help="Output JSONL path")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Randomly sample N items (default: all)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # Auto-download if not exists
    if not input_dir.exists() or not any(input_dir.rglob('*.json')):
        print(f"No data found at: {input_dir}")
        print("Cloning from GitHub: Yale-LILY/QMSum ...")
        import os
        repo_dir = input_dir.parent  # e.g. ../datasets/QMSum
        repo_dir.mkdir(parents=True, exist_ok=True)
        os.system(f"git clone https://github.com/Yale-LILY/QMSum.git {repo_dir}")
        if not input_dir.exists():
            print(f"Error: data dir still not found after clone: {input_dir}")
            sys.exit(1)
        print("Clone complete.")

    all_results = []

    for category in ['Product', 'Committee']:
        for split in ['train', 'val', 'test']:
            dir_path = Path(args.input_dir) / category / split
            if not dir_path.exists():
                continue
            for fp in sorted(dir_path.glob('*.json')):
                items = process_file(str(fp))
                all_results.extend(items)
                print(f"[{category}/{split}] {fp.name}: {len(items)} queries")

    total_before = len(all_results)

    if args.num_samples and args.num_samples < total_before:
        random.seed(42)
        all_results = random.sample(all_results, args.num_samples)
        print(f"\nRandomly sampled {args.num_samples} from {total_before} total")

    with open(args.output_path, 'w', encoding='utf-8') as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Total output: {len(all_results)} samples")
    print(f"Saved to: {args.output_path}")


if __name__ == '__main__':
    main()