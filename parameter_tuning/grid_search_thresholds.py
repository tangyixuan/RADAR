#!/usr/bin/env python3
"""Grid search for continuation and verdict thresholds.

The script sweeps over two values:
- Threshold A: difference between STOP and CONTINUE probabilities
- Threshold B: maximum label probability from the round judge

Both ranges are defined near the top of the file. Results are written as CSV/JSON
plus a summary of the best accuracy and macro-F1 settings.
"""

import argparse
import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing evaluation helpers that live under the top-level eval/ dir
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EVAL_PATH = os.path.join(PROJECT_ROOT, "eval")

if EVAL_PATH not in sys.path:
    sys.path.append(EVAL_PATH)

from eval import calculate_class_metrics  # noqa: E402

# Threshold search ranges (rounded to avoid floating point drift)
THRESHOLD_A_RANGE = np.round(np.arange(-1, 1.05, 0.05), 2)
THRESHOLD_B_RANGE = np.round(np.arange(0, 1.0, 0.05), 2)

LABELS = ["TRUE", "FALSE", "HALF-TRUE"]


def calculate_metrics_eval_style(y_true, y_pred):
    """Replicate the eval.py metric style (accuracy + macro-F1)."""
    if not y_true:
        return None

    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    metrics = {label: calculate_class_metrics(y_true, y_pred, label) for label in LABELS}
    macro_f1 = sum(f1 for _, _, f1 in metrics.values()) / len(LABELS)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "metrics": metrics,
    }


def load_json(file_path):
    """Load a UTF-8 JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_prediction_with_thresholds(
    claim_id,
    stop_data,
    label_data,
    threshold_a,
    threshold_b,
    start_round=1,
):
    """Return the predicted label for a claim under the given thresholds."""
    if claim_id not in stop_data or claim_id not in label_data:
        return None

    stop_decisions = stop_data[claim_id].get("continuation_decisions", [])

    # Iterate over round-level decisions, skipping index 0 which is usually metadata
    for round_idx, stop_decision in enumerate(stop_decisions[1:], start=start_round):
        stop_prob = stop_decision.get("choice_probabilities", {}).get("STOP", 0)
        continue_prob = stop_decision.get("choice_probabilities", {}).get("CONTINUE", 0)
        stop_continue_diff = stop_prob - continue_prob

        if stop_continue_diff >= threshold_a:
            # Judges can be serialized as judge_Xr or round_X_judge
            round_key_judge = f"judge_{round_idx}r"
            round_key_round = f"round_{round_idx}_judge"

            judge_data = None
            if round_key_judge in label_data[claim_id]:
                judge_data = label_data[claim_id][round_key_judge]
            elif round_key_round in label_data[claim_id]:
                judge_data = label_data[claim_id][round_key_round]

            if judge_data:
                verdict = judge_data.get("verdict")
                if verdict:
                    label_probs = judge_data.get("probability_info", {}).get("choice_probabilities", {})
                    if label_probs and verdict in label_probs:
                        verdict_prob = label_probs[verdict]
                        if verdict_prob >= threshold_b:
                            return verdict
                    elif not label_probs:
                        return verdict

    # Fall back to the final judge (or latest round judge) if no earlier rule matches
    if "final_judge" in label_data[claim_id]:
        final_judge_data = label_data[claim_id]["final_judge"]
        verdict = final_judge_data.get("verdict")
        if verdict:
            return verdict

    return None


def evaluate_thresholds(stop_data, label_data, ground_truth, threshold_a, threshold_b, start_round=1):
    """Compute metrics for a specific pair of thresholds."""
    predictions = {}

    for claim_id in ground_truth:
        pred = get_prediction_with_thresholds(
            claim_id,
            stop_data,
            label_data,
            threshold_a,
            threshold_b,
            start_round,
        )
        if pred:
            predictions[claim_id] = pred

    y_true, y_pred = [], []
    for claim_id, gt_label in ground_truth.items():
        if claim_id in predictions:
            y_true.append(gt_label)
            y_pred.append(predictions[claim_id])

    if not y_true:
        return None

    eval_results = calculate_metrics_eval_style(y_true, y_pred)
    if eval_results is None:
        return None

    results = {
        "threshold_a": round(threshold_a, 2),
        "threshold_b": round(threshold_b, 2),
        "accuracy": eval_results["accuracy"],
        "macro_f1": eval_results["macro_f1"],
        "num_predictions": len(y_true),
        "coverage": len(y_true) / len(ground_truth),
    }

    for label in LABELS:
        _, _, f1 = eval_results["metrics"][label]
        results[f"{label}_f1"] = f1

    return results


def run_grid_search(stop_file, label_file, gt_file, output_dir):
    """Execute the sweep over the pre-defined threshold grids."""
    start_round = 1
    stop_name = Path(stop_file).stem
    label_name = Path(label_file).stem
    output_name = f"round_{start_round}_{stop_name}_X_{label_name}"

    print(f"\nProcessing: {Path(stop_file).name} x {Path(label_file).name}")
    print(f"Output prefix: {output_name}")

    print("Loading data...")
    stop_data = load_json(stop_file)
    label_data = load_json(label_file)
    ground_truth = load_json(gt_file)

    print(f"Stop entries: {len(stop_data)} claims")
    print(f"Label entries: {len(label_data)} claims")
    print(f"Ground truth entries: {len(ground_truth)} claims")

    all_results = []
    total_combinations = len(THRESHOLD_A_RANGE) * len(THRESHOLD_B_RANGE)
    print(f"Evaluating {total_combinations} combinations...")

    for idx, (threshold_a, threshold_b) in enumerate(product(THRESHOLD_A_RANGE, THRESHOLD_B_RANGE), start=1):
        if idx % 10 == 0:
            print(f"Progress: {idx}/{total_combinations}")

        results = evaluate_thresholds(
            stop_data,
            label_data,
            ground_truth,
            threshold_a,
            threshold_b,
            start_round,
        )

        if results:
            all_results.append(results)

    if not all_results:
        print("WARNING: No valid results were produced.")
        print("Potential causes: invalid threshold ranges, data mismatch, or missing probabilities.")
        return None

    df = pd.DataFrame(all_results)

    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    if "/home/" in output_dir and os.path.exists("/Users"):
        output_dir = os.path.join(os.getcwd(), "DATA_NOW", "result")
        print(f"\nNotice: Output path adjusted to {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, f"results_{output_name}.csv")
    df.to_csv(output_csv, index=False)
    print("\nSaved grid search results:")
    print(f"  CSV:  {output_csv}")

    output_json_all = os.path.join(output_dir, f"results_{output_name}.json")
    with open(output_json_all, "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2, ensure_ascii=False)
    print(f"  JSON: {output_json_all}")

    best_acc_idx = df["accuracy"].idxmax()
    best_f1_idx = df["macro_f1"].idxmax()

    print("\n" + "=" * 80)
    print("=== Best Accuracy ===")
    print("=" * 80)
    print(df.iloc[best_acc_idx].to_string())

    print("\n" + "=" * 80)
    print("=== Best Macro F1 ===")
    print("=" * 80)
    print(df.iloc[best_f1_idx].to_string())

    best_results = {
        "best_accuracy": df.iloc[best_acc_idx].to_dict(),
        "best_macro_f1": df.iloc[best_f1_idx].to_dict(),
    }

    output_json = os.path.join(output_dir, f"best_results_{output_name}.json")
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(best_results, file, indent=2, ensure_ascii=False)
    print(f"\nSaved summary: {output_json}")

    return df


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Grid search over STOP/CONTINUE and label-confidence thresholds.\n\n"
            "Threshold definitions:\n"
            "  • Threshold A = stop_prob - continue_prob\n"
            "  • Threshold B = highest label probability\n"
            "Ranges are configured near the top of the script."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python grid_search_thresholds.py \\\n                --stop_file stop_or_not/matching_evidence_answer_map_multi_people_bj_llama.json \\\n                --label_file label_threshold/matching_evidence_answer_map_multi_people_round_judges_llama.json\n"
            "  python grid_search_thresholds.py \\\n                --stop_file stop_or_not/matching_evidence_full_answer_map_multi_people_continue_check_llama.json \\\n                --label_file label_threshold/matching_evidence_full_answer_map_multi_people_round_judges_llama.json \\\n                --output_dir result\n"
        ),
    )

    parser.add_argument(
        "--stop_file",
        type=str,
        required=True,
        help="Path to the STOP/CONTINUE probability JSON",
    )
    parser.add_argument(
        "--label_file",
        type=str,
        required=True,
        help="Path to the per-round label JSON",
    )
    parser.add_argument(
        "--gt_file",
        type=str,
        default="./data/GT_test_all.json",
        help="Ground-truth labels JSON (default: ./data/GT_test_all.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/parameter_tuning_results",
    )

    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()

    missing = []
    for label, path in ("Stop", args.stop_file), ("Label", args.label_file), ("Ground truth", args.gt_file):
        if not os.path.exists(path):
            missing.append((label, path))
    if missing:
        for label, path in missing:
            print(f"ERROR: {label} file not found: {path}")
        return

    print("=" * 80)
    print("Starting grid search...")
    print("=" * 80)
    print(f"Stop file:       {args.stop_file}")
    print(f"Label file:      {args.label_file}")
    print(f"Ground truth:    {args.gt_file}")
    print(f"Output dir:      {args.output_dir}")
    print(f"Threshold A span: {THRESHOLD_A_RANGE.min():.2f} to {THRESHOLD_A_RANGE.max():.2f}")
    print(f"Threshold B span: {THRESHOLD_B_RANGE.min():.2f} to {THRESHOLD_B_RANGE.max():.2f}")
    print("=" * 80)

    df = run_grid_search(args.stop_file, args.label_file, args.gt_file, args.output_dir)

    if df is None:
        print("\n" + "=" * 80)
        print("Grid search failed.")
        print("=" * 80)
        return

    print("\n" + "=" * 80)
    print("Grid search complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
