#!/bin/bash
# Run multi_people mode with Qwen model on feverous_golden_evidence_new2.json

echo "=========================================="
echo "Running multi_people with Qwen"
echo "Dataset: feverous_golden_evidence_new2.json"
echo "=========================================="
echo ""

python main.py \
    --mode multi_people \
    --model qwen \
    --input_file /home/qqs/mad_formal_0916/data/feverous_golden_evidence_new2.json

echo ""
echo "=========================================="
echo "✅ Completed!"
echo "=========================================="

