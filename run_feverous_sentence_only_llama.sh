#!/bin/bash
# Run multi_people mode with LLaMA model on feverous_golden_evidence_new.json

echo "=========================================="
echo "Running multi_people with LLaMA"
echo "Dataset: feverous_golden_evidence_new.json"
echo "=========================================="
echo ""

python main.py \
    --mode multi_people \
    --model llama \
    --input_file /home/qqs/mad_formal_0916/data/feverous_golden_evidence_new.json

echo ""
echo "=========================================="
echo "✅ Completed!"
echo "=========================================="

