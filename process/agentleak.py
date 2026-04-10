#!/usr/bin/env python3
"""
Process AgentLeak dataset into JSONL format for gpt_tagger.py

prompt = query (objective.user_request) + context (privacy_instruction + private_vault.records)

Usage:
    python agentleak.py <datasets_dir> <output_dir> [--num_samples N]

Example:
    python agentleak.py ../datasets/AgentLeak/agentleak_data/datasets ./processed_data --num_samples 10

Auto-downloads from GitHub if local path doesn't exist.
"""

import json
import random
import argparse
import os
from pathlib import Path


def auto_download(datasets_dir):
    """Clone repo if local path doesn't exist."""
    datasets_dir = Path(datasets_dir)
    if datasets_dir.exists():
        return

    # Guess repo root (2 levels up from agentleak_data/datasets)
    repo_dir = datasets_dir.parent.parent
    print(f"Local path not found: {datasets_dir}")
    print("Cloning from GitHub: Privatris/AgentLeak ...")
    repo_dir.mkdir(parents=True, exist_ok=True)
    os.system(f"git clone https://github.com/Privatris/AgentLeak.git {repo_dir}")
    if not datasets_dir.exists():
        print(f"Error: datasets dir still not found after clone: {datasets_dir}")
        exit(1)
    print("Clone complete.")


def format_records(records):
    """Format private_vault.records into readable text."""
    lines = []
    for rec in records:
        record_type = rec.get("record_type", "unknown")
        fields = rec.get("fields", {})
        parts = [f"[{record_type}]"]
        for k, v in fields.items():
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v)
            parts.append(f"  {k}: {v}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


def process_file(filepath):
    """Process a single JSONL file."""
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            objective = data.get("objective", {})
            user_request = objective.get("user_request", "")

            vault = data.get("private_vault", {})
            records = vault.get("records", [])
            records_text = format_records(records)

            prompt = f"{user_request}\n\nContext:\n{records_text}"

            results.append({"prompt": prompt})

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Path to AgentLeak/agentleak_data/datasets directory")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Randomly sample N items (default: all)")
    args = parser.parse_args()

    auto_download(args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    input_dir = Path(args.input_dir)

    # Only process scenarios_full_1000.jsonl
    target_file = input_dir / "scenarios_full_1000.jsonl"
    if not target_file.exists():
        print(f"Error: {target_file} not found")
        exit(1)

    all_results = process_file(str(target_file))
    print(f"scenarios_full_1000.jsonl: {len(all_results)} samples")

    total_before = len(all_results)

    if args.num_samples and args.num_samples < total_before:
        random.seed(42)
        all_results = random.sample(all_results, args.num_samples)
        print(f"\nRandomly sampled {args.num_samples} from {total_before} total")

    output_path = os.path.join(args.output_dir, "agentleak.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Total output: {len(all_results)} samples")
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()