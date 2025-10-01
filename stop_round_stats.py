#!/usr/bin/env python3
"""Compute stop-round distributions for multi-agent evidence files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

DEFAULT_FILES = [
    "data/matching_evidence_answer_map_multi_people_bj_llama.json",
    "data/matching_evidence_answer_map_multi_people_fj_llama.json",
    "data/matching_evidence_full_answer_map_multi_people_bj_llama.json",
    "data/matching_evidence_full_answer_map_multi_people_fj_llama.json",
]

ROUND_ORDER = ["opening", "rebuttal", "closing", "no_stop", "unknown"]
VERDICT_FOCUS = ["TRUE", "HALF-TRUE", "FALSE"]


def extract_stop_round(decisions: Iterable[Dict]) -> str:
    """Return the round where the first stop occurs, or 'no_stop'."""
    for decision in decisions or []:
        if str(decision.get("decision", "")).lower() == "stop":
            return str(decision.get("round", "unknown") or "unknown")
    return "no_stop"


def extract_verdict(raw: str) -> str:
    """Pull the verdict label from the raw verdict string."""
    if not raw:
        return "UNKNOWN"
    text = str(raw)
    for marker in ("VERDICT:", "[VERDICT]:", "Verdict:", "verdict:"):
        if marker in text:
            text = text.split(marker, 1)[1]
            break
    text = text.strip()
    if not text:
        return "UNKNOWN"
    return text.splitlines()[0].strip()


def normalize_verdict(verdict: str) -> str:
    token = verdict.upper().replace("_", "-").replace(" ", "")
    if token in {"TRUE"}:
        return "TRUE"
    if token in {"HALF-TRUE", "HALFTRUE"}:
        return "HALF-TRUE"
    if token in {"FALSE"}:
        return "FALSE"
    return "OTHER"


def order_keys(counter: Counter) -> Iterable[str]:
    seen = set()
    for key in ROUND_ORDER:
        if counter.get(key):
            seen.add(key)
            yield key
    for key in sorted(counter):
        if key not in seen:
            yield key


def format_distribution(counter: Counter, total: int) -> Iterable[str]:
    for key in order_keys(counter):
        count = counter[key]
        pct = (count / total * 100) if total else 0
        yield f"  - {key}: {count} ({pct:.1f}%)"


def summarize_file(path: Path) -> Tuple[int, Counter, Counter, Dict[str, Counter]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    stop_counts: Counter = Counter()
    verdict_counts: Counter = Counter()
    verdict_stop_counts: Dict[str, Counter] = defaultdict(Counter)

    for item in data.values():
        stop_round = extract_stop_round(item.get("continuation_decisions", []))
        stop_counts[stop_round] += 1

        verdict_raw = item.get("final_verdict", "")
        verdict_label = normalize_verdict(extract_verdict(verdict_raw))
        verdict_counts[verdict_label] += 1
        verdict_stop_counts[verdict_label][stop_round] += 1

    return len(data), stop_counts, verdict_counts, verdict_stop_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute stop-round distributions for multi-agent evidence files."
    )
    parser.add_argument(
        "files",
        nargs="*",
        default=DEFAULT_FILES,
        help="Paths to JSON files to analyse (defaults to the four llama outputs).",
    )
    args = parser.parse_args()

    for idx, file_arg in enumerate(args.files):
        path = Path(file_arg)
        if not path.exists():
            print(f"\n{path}: missing file")
            continue

        total, stop_counts, verdict_counts, verdict_stop_counts = summarize_file(path)

        print(f"\nFile: {path}")
        print(f"Total cases: {total}")
        print("Stop-round distribution:")
        for line in format_distribution(stop_counts, total):
            print(line)

        for verdict in VERDICT_FOCUS:
            verdict_total = verdict_counts.get(verdict, 0)
            if verdict_total == 0:
                continue
            print(f"{verdict} stop distribution (n={verdict_total}):")
            for line in format_distribution(verdict_stop_counts[verdict], verdict_total):
                print(line)

        other_total = verdict_counts.get("OTHER", 0)
        if other_total:
            print(f"OTHER verdicts present: {other_total} (not broken down above)")

        if idx + 1 < len(args.files):
            print()


if __name__ == "__main__":
    main()
