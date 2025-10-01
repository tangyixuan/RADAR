import os
import json
import re
import sys
import argparse
from collections import defaultdict

def calculate_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def calculate_class_metrics(y_true, y_pred, class_label):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == p == class_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != class_label and p == class_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == class_label and p != class_label)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = calculate_f1_score(precision, recall)
    return precision, recall, f1

def extract_round_verdicts(pred_file):
    """Extract verdicts from round 1, round 2, and final judge"""
    with open(pred_file, "r") as f:
        data = json.load(f)
    
    round_verdicts = {
        'round_1': {},
        'round_2': {},
        'final': {}
    }
    
    probabilities = {
        'round_1': {},
        'round_2': {},
        'final': {}
    }
    
    for example_id, content in data.items():
        # Extract Round 1 verdict
        if 'round_1_judge' in content and isinstance(content['round_1_judge'], dict):
            verdict = content['round_1_judge'].get('verdict', 'UNKNOWN')
            prob = content['round_1_judge'].get('probability', None)
            round_verdicts['round_1'][example_id] = verdict
            probabilities['round_1'][example_id] = prob
        
        # Extract Round 2 verdict
        if 'round_2_judge' in content and isinstance(content['round_2_judge'], dict):
            verdict = content['round_2_judge'].get('verdict', 'UNKNOWN')
            prob = content['round_2_judge'].get('probability', None)
            round_verdicts['round_2'][example_id] = verdict
            probabilities['round_2'][example_id] = prob
        
        # Extract Final verdict
        if 'final_judge' in content and isinstance(content['final_judge'], dict):
            verdict = content['final_judge'].get('verdict', 'UNKNOWN')
            prob = content['final_judge'].get('probability', None)
            round_verdicts['final'][example_id] = verdict
            probabilities['final'][example_id] = prob
    
    return round_verdicts, probabilities

def evaluate_round(round_name, predictions, groundtruth, probabilities=None):
    """Evaluate a single round's predictions"""
    total, correct = 0, 0
    label_count = defaultdict(int)
    correct_count = defaultdict(int)
    y_true, y_pred = [], []
    
    # Statistics for probabilities
    prob_stats = {
        'has_prob': 0,
        'no_prob': 0,
        'avg_prob': None,
        'avg_prob_by_label': {}
    }
    
    prob_values = []
    prob_by_label = defaultdict(list)
    
    for key in groundtruth:
        if key in predictions:
            total += 1
            true_label = groundtruth[key]
            pred_label = predictions[key]
            y_true.append(true_label)
            y_pred.append(pred_label)
            label_count[true_label] += 1
            
            # Track probabilities
            if probabilities and key in probabilities:
                prob = probabilities[key]
                if prob is not None:
                    prob_stats['has_prob'] += 1
                    prob_values.append(prob)
                    prob_by_label[pred_label].append(prob)
                else:
                    prob_stats['no_prob'] += 1
            
            if true_label == pred_label:
                correct += 1
                correct_count[true_label] += 1
    
    # Calculate average probabilities
    if prob_values:
        prob_stats['avg_prob'] = sum(prob_values) / len(prob_values)
        for label in prob_by_label:
            prob_stats['avg_prob_by_label'][label] = sum(prob_by_label[label]) / len(prob_by_label[label])
    
    accuracy = correct / total if total > 0 else 0.0
    metrics = {label: calculate_class_metrics(y_true, y_pred, label) 
               for label in ["TRUE", "HALF-TRUE", "FALSE"]}
    macro_f1 = sum(f1 for _, _, f1 in metrics.values()) / 3
    
    print(f"\n  === {round_name} ===")
    print(f"  Total examples: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Overall Accuracy: {accuracy:.2%}")
    
    # Print probability statistics
    if probabilities:
        print(f"\n  Probability Statistics:")
        print(f"    Examples with probability: {prob_stats['has_prob']}")
        print(f"    Examples without probability: {prob_stats['no_prob']}")
        if prob_stats['avg_prob'] is not None:
            print(f"    Average probability: {prob_stats['avg_prob']:.4f}")
            print(f"    Average probability by predicted label:")
            for label in ["TRUE", "HALF-TRUE", "FALSE"]:
                if label in prob_stats['avg_prob_by_label']:
                    print(f"      {label}: {prob_stats['avg_prob_by_label'][label]:.4f}")
    
    print(f"\n  Class-wise Accuracy:")
    for label in ["TRUE", "HALF-TRUE", "FALSE"]:
        correct_num = correct_count[label]
        total_num = label_count[label]
        acc = correct_num / total_num if total_num > 0 else 0.0
        print(f"    {label}: {acc:.2%} ({correct_num}/{total_num})")
    
    print(f"\n  F1 Scores:")
    for label in ["TRUE", "HALF-TRUE", "FALSE"]:
        p, r, f1 = metrics[label]
        print(f"    {label} - Precision: {p:.2%}, Recall: {r:.2%}, F1: {f1:.2%}")
    print(f"  Macro-F1: {macro_f1:.2%}")
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'total': total,
        'correct': correct,
        'metrics': metrics,
        'prob_stats': prob_stats
    }

def evaluate(pred_file, groundtruth_file):
    """Evaluate all rounds"""
    round_verdicts, probabilities = extract_round_verdicts(pred_file)
    
    with open(groundtruth_file, "r") as f:
        groundtruth = json.load(f)
    
    print(f"\nFile: {pred_file}")
    print(f"Total examples in file: {len(round_verdicts['round_1'])}")
    print("=" * 80)
    
    results = {}
    for round_name in ['round_1', 'round_2', 'final']:
        round_label = {
            'round_1': 'Round 1 Judge (After Opening)',
            'round_2': 'Round 2 Judge (After Rebuttal)',
            'final': 'Final Judge (After Closing)'
        }[round_name]
        
        results[round_name] = evaluate_round(
            round_label, 
            round_verdicts[round_name], 
            groundtruth,
            probabilities[round_name]
        )
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(f"{'Round':<30} {'Accuracy':<12} {'Macro-F1':<12} {'With Prob':<15}")
    print("-" * 80)
    for round_name in ['round_1', 'round_2', 'final']:
        round_label = {
            'round_1': 'Round 1 (Opening)',
            'round_2': 'Round 2 (Rebuttal)',
            'final': 'Final (Closing)'
        }[round_name]
        
        acc = results[round_name]['accuracy']
        f1 = results[round_name]['macro_f1']
        has_prob = results[round_name]['prob_stats']['has_prob']
        total = results[round_name]['total']
        
        print(f"{round_label:<30} {acc:>10.2%}  {f1:>10.2%}  {has_prob}/{total}")
    
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    # Fixed groundtruth file path
    groundtruth_file = "../data/GT_test_all.json"
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate round-based prediction files')
    parser.add_argument('--prediction', nargs='+', required=True, 
                       help='Prediction file(s) to evaluate (can specify multiple files)')
    
    args = parser.parse_args()
    prediction_files = args.prediction
    
    print(f"Groundtruth file: {groundtruth_file}")
    print(f"Prediction files: {prediction_files}")
    print("=" * 80)
    
    for pred_file in prediction_files:
        if not os.path.exists(pred_file):
            print(f"Warning: Prediction file {pred_file} does not exist, skipping...")
            continue
        evaluate(pred_file, groundtruth_file)
        print("\n\n")
