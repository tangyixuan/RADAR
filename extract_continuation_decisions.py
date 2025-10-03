#!/usr/bin/env python3
"""Utility to extract continuation decisions and probabilities per claim."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the final decision and choice probabilities for each continuation decision round."
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="JSON files containing continuation decision data",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional CSV file to write the aggregated results. Defaults to stdout.",
    )
    return parser.parse_args()


def iter_decisions(path: Path) -> Iterator[Dict[str, object]]:
    with path.open() as handle:
        payload = json.load(handle)

    for claim_id, claim_payload in payload.items():
        for decision in claim_payload.get("continuation_decisions", []):
            choice_probs = decision.get("choice_probabilities", {})
            yield {
                "source_file": str(path),
                "claim_id": claim_id,
                "round": decision.get("round", ""),
                "decision": decision.get("decision", ""),
                "prob_stop": choice_probs.get("STOP"),
                "prob_continue": choice_probs.get("CONTINUE"),
            }


def write_rows(rows: Iterable[Dict[str, object]], output: Path | None) -> None:
    fieldnames: List[str] = [
        "source_file",
        "claim_id",
        "round",
        "decision",
        "prob_stop",
        "prob_continue",
    ]

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        destination = output.open("w", newline="")
        sink = destination
    else:
        destination = None
        sink = sys.stdout

    writer = csv.DictWriter(sink, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    if destination is not None:
        destination.close()


def main() -> None:
    args = parse_args()

    rows = (row for file_path in args.files for row in iter_decisions(file_path))
    write_rows(rows, args.output)


if __name__ == "__main__":
    try:
        main()
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"Failed to parse JSON: {exc}\n")
        sys.exit(1)
