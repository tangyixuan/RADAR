#!/bin/bash

# run_adaptive_early_stopping_experiments.sh
# Script to run experiments with different adaptive early stopping thresholds

echo "======================================================================"
echo "ADAPTIVE EARLY STOPPING EXPERIMENTS"
echo "Testing different threshold configurations for multi-agent debate"
echo "======================================================================"

# Base parameters
MODEL_NAME="llama"  # Change to your model
INPUT_FILE="data1/retrieved_evidence_bgebase_answer_map_multi_people_3_llama.json"
OUTPUT_BASE="data1/retrieved_evidence_bgebase_answer_map_multi_people_hybrid_adaptive"

echo "Running experiments with different threshold configurations..."

# Experiment 1: Conservative thresholds (thorough debate)
echo ""
echo "🔹 EXPERIMENT 1: Conservative Thresholds (tau_s=1.0, tau_v=0.8)"
echo "   Expected: More complete debates, fewer early stops"
python main.py \
    --input_file "$INPUT_FILE" \
    --output_file "${OUTPUT_BASE}_conservative_${MODEL_NAME}.json" \
    --agent_type "multi_agent_people_hybrid_adaptive" \
    --model_name "$MODEL_NAME" \
    --tau_s 1.0 \
    --tau_v 0.8 \
    --max_samples 50

# Experiment 2: Balanced thresholds (default)  
echo ""
echo "🔹 EXPERIMENT 2: Balanced Thresholds (tau_s=0.5, tau_v=0.7)"
echo "   Expected: Good balance of efficiency and thoroughness"
python main.py \
    --input_file "$INPUT_FILE" \
    --output_file "${OUTPUT_BASE}_balanced_${MODEL_NAME}.json" \
    --agent_type "multi_agent_people_hybrid_adaptive" \
    --model_name "$MODEL_NAME" \
    --tau_s 0.5 \
    --tau_v 0.7 \
    --max_samples 50

# Experiment 3: Aggressive thresholds (fast decisions)
echo ""
echo "🔹 EXPERIMENT 3: Aggressive Thresholds (tau_s=0.0, tau_v=0.6)" 
echo "   Expected: More early stops, faster processing"
python main.py \
    --input_file "$INPUT_FILE" \
    --output_file "${OUTPUT_BASE}_aggressive_${MODEL_NAME}.json" \
    --agent_type "multi_agent_people_hybrid_adaptive" \
    --model_name "$MODEL_NAME" \
    --tau_s 0.0 \
    --tau_v 0.6 \
    --max_samples 50

# Experiment 4: High confidence threshold (quality focus)
echo ""
echo "🔹 EXPERIMENT 4: High Confidence Threshold (tau_s=0.5, tau_v=0.85)"
echo "   Expected: Stops only with very high confidence"
python main.py \
    --input_file "$INPUT_FILE" \
    --output_file "${OUTPUT_BASE}_high_confidence_${MODEL_NAME}.json" \
    --agent_type "multi_agent_people_hybrid_adaptive" \
    --model_name "$MODEL_NAME" \
    --tau_s 0.5 \
    --tau_v 0.85 \
    --max_samples 50

echo ""
echo "======================================================================"
echo "EXPERIMENT ANALYSIS"
echo "======================================================================"

# Run analysis script to compare results
python -c "
import json
import sys
sys.path.append('.')
from agents.multi_agent_people_hybrid import analyze_early_stopping_performance

# Load and analyze all experiment results
experiments = [
    ('Conservative', '${OUTPUT_BASE}_conservative_${MODEL_NAME}.json'),
    ('Balanced', '${OUTPUT_BASE}_balanced_${MODEL_NAME}.json'), 
    ('Aggressive', '${OUTPUT_BASE}_aggressive_${MODEL_NAME}.json'),
    ('High Confidence', '${OUTPUT_BASE}_high_confidence_${MODEL_NAME}.json')
]

print('\\nEARLY STOPPING PERFORMANCE COMPARISON')
print('=' * 60)

for name, filename in experiments:
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        total_cases = len(data)
        early_stops = 0
        total_decisions = 0
        avg_stop_margin = 0
        avg_confidence = 0
        
        for case_id, case_data in data.items():
            decisions = case_data.get('continuation_decisions', [])
            total_decisions += len(decisions)
            
            for decision in decisions:
                if decision.get('terminated_early', False):
                    early_stops += 1
                if decision.get('stop_margin') is not None:
                    avg_stop_margin += decision['stop_margin']
                if decision.get('confidence') is not None:
                    avg_confidence += decision['confidence']
        
        early_stop_rate = early_stops / total_decisions if total_decisions > 0 else 0
        avg_stop_margin = avg_stop_margin / total_decisions if total_decisions > 0 else 0
        avg_confidence = avg_confidence / total_decisions if total_decisions > 0 else 0
        
        print(f'\\n{name}:')
        print(f'  Cases processed: {total_cases}')
        print(f'  Total decisions: {total_decisions}')
        print(f'  Early stops: {early_stops} ({early_stop_rate:.1%})')
        print(f'  Avg stop margin: {avg_stop_margin:.3f}')
        print(f'  Avg confidence: {avg_confidence:.3f}')
        
    except FileNotFoundError:
        print(f'\\n{name}: File not found - {filename}')
    except Exception as e:
        print(f'\\n{name}: Error - {str(e)}')

print('\\n' + '=' * 60)
print('INTERPRETATION GUIDE:')
print('• Early stop rate: Higher = more efficient, lower = more thorough')
print('• Stop margin: Positive = preference for stopping')
print('• Confidence: Higher = more certain verdicts')
print('• Conservative config should have lowest early stop rate')
print('• Aggressive config should have highest early stop rate')
"

echo ""
echo "======================================================================"
echo "RECOMMENDATIONS"
echo "======================================================================"
echo ""
echo "Based on your results:"
echo "• If early stop rate is too high → Increase tau_s and/or tau_v"
echo "• If early stop rate is too low → Decrease tau_s and/or tau_v" 
echo "• For production use, balanced thresholds (0.5, 0.7) are recommended"
echo "• Monitor confidence levels to ensure verdict quality"
echo ""
echo "Experiment completed! Check output files for detailed results."
