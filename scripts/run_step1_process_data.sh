#!/bin/bash
# Step 1: Process raw datasets into unified processed_data/*.jsonl
#
# Usage:
#   bash run_step1_process_data.sh          # default 1 sample per dataset (smoke test)
#
# For a real run, set NUM_SAMPLES=1000 (see README.md for the resulting row counts).
#
# Sampling is random, not first-N: each process/*.py script picks rows via
# random.sample() with a fixed random.seed(42), so results are reproducible
# across re-runs but not a positional slice of the source data.

set -e

NUM_SAMPLES=1                  # <-- set number per dataset; 1000 for a real run, or "all" for full data
OUTPUT_DIR="../processed_data"

mkdir -p "$OUTPUT_DIR"

EXISTING=$(ls "$OUTPUT_DIR"/*.jsonl 2>/dev/null | wc -l)

if [ "$EXISTING" -gt 0 ]; then
    echo "=== Found $EXISTING JSONL files in $OUTPUT_DIR, skipping data processing ==="
    echo "    (Delete $OUTPUT_DIR to re-process everything, or run a single"
    echo "     process/<dataset>.py script directly to backfill one dataset.)"
    exit 0
fi

# Build --num_samples flag
if [ "$NUM_SAMPLES" = "all" ]; then
    SAMPLE_FLAG=""
    echo "=== Processing ALL data ==="
else
    SAMPLE_FLAG="--num_samples $NUM_SAMPLES"
    echo "=== Sampling $NUM_SAMPLES per dataset ==="
fi

echo ""

# --- 1. QMSum (Product + Committee) ---
echo "[1/6] Processing QMSum..."
python ../process/qmsum.py ../datasets/QMSum/data "$OUTPUT_DIR/qmsum_product_committee.jsonl" $SAMPLE_FLAG
echo ""

# --- 2. Fortress (benign + risky) ---
echo "[2/6] Processing Fortress..."
python ../process/fortress.py ../datasets/fortress_public "$OUTPUT_DIR" $SAMPLE_FLAG
echo ""

# --- 3. ConfAIde (tier 1, 2, 3) ---
echo "[3/6] Processing ConfAIde..."
python ../process/confaide.py ../datasets/confaide/benchmark "$OUTPUT_DIR" $SAMPLE_FLAG
echo ""

# --- 4. AgentLeak ---
echo "[4/6] Processing AgentLeak..."
python ../process/agentleak.py ../datasets/AgentLeak/agentleak_data/datasets "$OUTPUT_DIR" $SAMPLE_FLAG
echo ""

# --- Sensitive Document Classification ---
# Skipped by default: the source dataset mouhamet/sensitive_document_classification
# has been removed from HuggingFace (confirmed 404 "Repository not found" even with
# an authenticated token, not a gating/permission issue). Uncomment if it ever
# reappears under this name or a mirror is found.
# echo "Processing Sensitive Document Classification..."
# python ../process/sensitive_doc.py ../datasets/sensitive_document_classification "$OUTPUT_DIR" $SAMPLE_FLAG
# echo ""

# --- 5. Confidential Business Excerpts ---
echo "[5/6] Processing Confidential Business Excerpts..."
python ../process/confidential_biz.py ../datasets/confidential_biz "$OUTPUT_DIR" $SAMPLE_FLAG
echo ""

# --- 6. PKU-SafeRLHF (Endangering National Security) ---
echo "[6/6] Processing PKU-SafeRLHF..."
python ../process/pku_saferlhf.py ../datasets/pku_saferlhf "$OUTPUT_DIR" $SAMPLE_FLAG
echo ""

# --- GovReport Summarization ---
# Skipped by default: GovReport summaries are already-public GAO/CRS
# reports, so they mostly yield false-positive "sensitive" tags rather
# than genuine policy violations. Uncomment if needed as negatives.
# echo "Processing GovReport..."
# python ../process/govreport.py ../datasets/govreport "$OUTPUT_DIR" $SAMPLE_FLAG
# echo ""

# --- US Business Data ---
# Skipped by default: same reasoning as GovReport — public company
# marketing descriptions, low positive-tag yield. Uncomment if needed.
# echo "Processing US Business Data..."
# python ../process/us_bizdata.py ../datasets/us_bizdata "$OUTPUT_DIR" $SAMPLE_FLAG
# echo ""

echo "=========================================="
echo "Done! All outputs in: $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "(no outputs yet)"
echo "=========================================="
