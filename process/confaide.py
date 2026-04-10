#!/usr/bin/env python3
"""
Process ConfAIde benchmark (tier 1, 2, 3) into JSONL format for gpt_tagger.py

Tier 1: Extract "Information: ..." part as prompt
Tier 2: Use full line (tier_2a + tier_2b) as prompt
Tier 3: Each BEGIN/END block = context, paired with corresponding tier_3_control query

Usage:
    python confaide.py <benchmark_dir> <output_dir> [--num_samples N]

Example:
    python confaide.py ../datasets/confaide/benchmark ./processed_data --num_samples 10

Auto-downloads from GitHub if local path doesn't exist.
"""

import json
import re
import random
import argparse
import os
from pathlib import Path


def auto_download(benchmark_dir):
    """Clone repo if local path doesn't exist."""
    benchmark_dir = Path(benchmark_dir)
    if benchmark_dir.exists():
        return

    repo_dir = benchmark_dir.parent
    print(f"Local path not found: {benchmark_dir}")
    print("Cloning from GitHub: skywalker023/confaide ...")
    repo_dir.mkdir(parents=True, exist_ok=True)
    os.system(f"git clone https://github.com/skywalker023/confaide.git {repo_dir}")
    if not benchmark_dir.exists():
        print(f"Error: benchmark dir still not found after clone: {benchmark_dir}")
        exit(1)
    print("Clone complete.")


def parse_tier1(filepath):
    """Extract 'Information: ...' part from each line."""
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Find "Information: ..." at the end
            match = re.search(r'Information:\s*(.+)$', line)
            if match:
                info = match.group(1).strip()
                results.append({"prompt": info})
            else:
                results.append({"prompt": line})
    return results


def parse_tier2(filepath):
    """Extract content after \\n delimiter (skip the rating instruction prefix)."""
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split on literal \\n or actual \n after the instruction prefix
            if '\\n' in line:
                parts = line.split('\\n', 1)
                content = parts[1].strip() if len(parts) > 1 else line
            else:
                content = line
            if content:
                results.append({"prompt": content})
    return results


def parse_tier3(tier3_path, control_path):
    """Parse BEGIN/END blocks from tier_3.txt, pair with tier_3_control.txt queries."""
    # Read control queries
    with open(control_path, 'r', encoding='utf-8') as f:
        control_lines = [l.strip() for l in f.readlines() if l.strip()]

    # Parse BEGIN/END blocks
    with open(tier3_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = []
    pattern = r'<BEGIN>.*?\n(.*?)\n\n\n<END>'
    matches = re.findall(pattern, content, re.DOTALL)

    for m in matches:
        blocks.append(m.strip())

    results = []
    num_pairs = min(len(blocks), len(control_lines))

    for i in range(num_pairs):
        context = blocks[i]
        query = control_lines[i]
        prompt = f"{query}\n\nContext:\n{context}"
        results.append({"prompt": prompt})

    return results


def save_jsonl(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Path to confaide/benchmark directory")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Randomly sample N items per tier (default: all)")
    args = parser.parse_args()

    auto_download(args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    input_dir = Path(args.input_dir)

    # --- Tier 1 ---
    tier1 = parse_tier1(input_dir / "tier_1.txt")
    print(f"Tier 1: {len(tier1)} items")

    # --- Tier 2 (2a + 2b) ---
    tier2a = parse_tier2(input_dir / "tier_2a.txt")
    tier2b = parse_tier2(input_dir / "tier_2b.txt")
    tier2 = tier2a + tier2b
    print(f"Tier 2: {len(tier2)} items (2a={len(tier2a)}, 2b={len(tier2b)})")

    # --- Tier 3 ---
    tier3 = parse_tier3(input_dir / "tier_3.txt", input_dir / "tier_3_control.txt")
    print(f"Tier 3: {len(tier3)} items")

    # Combine all
    all_results = tier1 + tier2 + tier3
    print(f"Total: {len(all_results)} items")

    # Sample if needed
    if args.num_samples and args.num_samples < len(all_results):
        random.seed(42)
        all_results = random.sample(all_results, args.num_samples)
        print(f"Randomly sampled {args.num_samples}")

    output_path = os.path.join(args.output_dir, "confaide.jsonl")
    save_jsonl(all_results, output_path)
    print(f"Saved to: {output_path} ({len(all_results)} samples)")


if __name__ == '__main__':
    main()