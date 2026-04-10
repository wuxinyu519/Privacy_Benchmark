#!/bin/bash

# Usage:
#   bash run.sh          # default 10 samples per dataset

set -e

NUM_SAMPLES=10                 # <-- set number per dataset, or "all" for full data
OUTPUT_DIR="../processed_data"
TAGGED_DIR="../tagged_data"
API_KEY=""                     # <-- paste your OpenAI API key here
MODEL="gpt-4o"                 # <-- model to use

mkdir -p "$OUTPUT_DIR"

# ============================================================
# Step 1: Process datasets (skip if processed_data already has files)
# ============================================================
EXISTING=$(ls "$OUTPUT_DIR"/*.jsonl 2>/dev/null | wc -l)

if [ "$EXISTING" -gt 0 ]; then
    echo "=== Found $EXISTING JSONL files in $OUTPUT_DIR, skipping data processing ==="
    echo "    (Delete $OUTPUT_DIR to re-process)"
else
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
    echo "[1/9] Processing QMSum..."
    python ../process/qmsum.py ../datasets/QMSum/data "$OUTPUT_DIR/qmsum_product_committee.jsonl" $SAMPLE_FLAG
    echo ""

    # --- 2. Fortress (benign + risky) ---
    echo "[2/9] Processing Fortress..."
    python ../process/fortress.py ../datasets/fortress_public "$OUTPUT_DIR" $SAMPLE_FLAG
    echo ""

    # --- 3. ConfAIde (tier 1, 2, 3) ---
    echo "[3/9] Processing ConfAIde..."
    python ../process/confaide.py ../datasets/confaide/benchmark "$OUTPUT_DIR" $SAMPLE_FLAG
    echo ""

    # --- 4. AgentLeak ---
    echo "[4/9] Processing AgentLeak..."
    python ../process/agentleak.py ../datasets/AgentLeak/agentleak_data/datasets "$OUTPUT_DIR" $SAMPLE_FLAG
    echo ""

    # --- 5. Sensitive Document Classification ---
    echo "[5/9] Processing Sensitive Document Classification..."
    python ../process/sensitive_doc.py ../datasets/sensitive_document_classification "$OUTPUT_DIR" $SAMPLE_FLAG
    echo ""

    # --- 6. Confidential Business Excerpts ---
    echo "[6/9] Processing Confidential Business Excerpts..."
    python ../process/confidential_biz.py ../datasets/confidential_biz "$OUTPUT_DIR" $SAMPLE_FLAG
    echo ""

    # --- 7. PKU-SafeRLHF (Endangering National Security) ---
    echo "[7/9] Processing PKU-SafeRLHF..."
    python ../process/pku_saferlhf.py ../datasets/pku_saferlhf "$OUTPUT_DIR" $SAMPLE_FLAG
    echo ""

    # --- 8. GovReport Summarization ---
    echo "[8/9] Processing GovReport..."
    python ../process/govreport.py ../datasets/govreport "$OUTPUT_DIR" $SAMPLE_FLAG
    echo ""

    # --- 9. US Business Data ---
    echo "[9/9] Processing US Business Data..."
    python ../process/us_bizdata.py ../datasets/us_bizdata "$OUTPUT_DIR" $SAMPLE_FLAG
    echo ""

    echo "=========================================="
    echo "Done! All outputs in: $OUTPUT_DIR/"
    ls -lh "$OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "(no outputs yet)"
    echo "=========================================="
fi

# ============================================================
# Step 2: GPT Tagger Inference
# ============================================================
echo ""
echo "=========================================="
echo "[GPT Tagger] Starting inference..."
echo "  Input:  $OUTPUT_DIR"
echo "  Output: $TAGGED_DIR"
echo "  Model:  $MODEL"
echo "=========================================="

python ../gpt_infer/gpt_tagger.py "$OUTPUT_DIR" "$API_KEY" "$NUM_SAMPLES" "$TAGGED_DIR" "$MODEL"

echo "=========================================="
echo "Tagging done! Tagged outputs in: $TAGGED_DIR/"
ls -lh "$TAGGED_DIR"/*.jsonl 2>/dev/null || echo "(no tagged outputs yet)"
echo "=========================================="