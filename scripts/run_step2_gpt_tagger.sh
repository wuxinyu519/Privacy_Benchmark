#!/bin/bash
# Step 2: GPT tagger inference over processed_data/*.jsonl, grounded in
# RedacBench policies (see ../gpt_infer/gpt_tagger.py).
#
# Usage:
#   bash run_step2_gpt_tagger.sh
#
# Run this after run_step1_process_data.sh has populated ../processed_data/.
# Default is 5 (smoke test / prompt-quality check). For a real run, set
# NUM_SAMPLES=1000 (see README.md for the resulting row counts).

set -e

NUM_SAMPLES=5                  # <-- how many per file to tag; 1000 for a real run, or "all"
OUTPUT_DIR="../processed_data"                          # <-- input, produced by step 1
TAGGED_DIR="../tagged_data"
API_KEY="${OPENAI_API_KEY:-}"  # <-- reads from env var; `export OPENAI_API_KEY=sk-...` before running (never hardcode a real key here — this file is committed to git)
MODEL="gpt-5.5"                 # <-- model to use
POLICY_DATASET="../redacBench/benchmark_dataset.json"   # <-- RedacBench policy list used to ground tags

echo "=========================================="
echo "[GPT Tagger] Starting inference..."
echo "  Input:    $OUTPUT_DIR"
echo "  Output:   $TAGGED_DIR"
echo "  Model:    $MODEL"
echo "  Policies: $POLICY_DATASET"
echo "=========================================="

python ../gpt_infer/gpt_tagger.py "$OUTPUT_DIR" "$API_KEY" "$NUM_SAMPLES" "$TAGGED_DIR" "$MODEL" "$POLICY_DATASET"

echo "=========================================="
echo "Tagging done! Tagged outputs in: $TAGGED_DIR/"
ls -lh "$TAGGED_DIR"/*.jsonl 2>/dev/null || echo "(no tagged outputs yet)"
echo "=========================================="
