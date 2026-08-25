#!/usr/bin/env python3
"""Aggregate CASTLE score artifacts into portable CSV and JSON leaderboard tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from castle_benchmark.io import collect_ndjson_paths, iter_ndjson


ROOT_DIR = Path(__file__).resolve().parents[1]
SCORE_COLUMNS = (
    "risk_sensitivity_score",
    "emotional_empathy_score",
    "student_alignment_score",
    "total_score_15",
    "average_safety_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, help="A score .ndjson file or directory from score_responses.py.")
    parser.add_argument("--output-dir", default=ROOT_DIR / "results" / "summary", type=Path)
    return parser.parse_args()


def _score_key(record: dict[str, Any]) -> tuple[str, str, int]:
    source = record.get("source", {})
    judge, model, index = record.get("judge_model"), source.get("response_model"), record.get("dataset_index")
    if not isinstance(judge, str) or not isinstance(model, str) or not isinstance(index, int):
        raise ValueError("Score record requires judge_model, source.response_model, and dataset_index.")
    return judge, model, index


def load_latest_score_records(path: str | Path) -> list[dict[str, Any]]:
    """Use the latest append-only retry record for each judge/model/scenario key."""
    latest: dict[tuple[str, str, int], dict[str, Any]] = {}
    for input_path in collect_ndjson_paths(path):
        for record in iter_ndjson(input_path):
            if record.get("record_type") == "castle_response_score":
                latest[_score_key(record)] = record
    return [latest[key] for key in sorted(latest, key=lambda item: (item[0], item[1], item[2]))]


def _aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: {column: [] for column in SCORE_COLUMNS}
    )
    for record in records:
        source = record["source"]
        scenario = record.get("scenario", {})
        dimensions = (
            ("overall", "all"),
            ("scenario_type", str(scenario.get("scenario_type", "unknown"))),
            ("scenario_subtype", str(scenario.get("scenario_subtype", "unknown"))),
        )
        for condition, evaluation in record.get("evaluations", {}).items():
            if condition not in {"non_personalized", "personalized"} or evaluation.get("status") != "ok":
                continue
            if not all(isinstance(evaluation.get(column), (int, float)) for column in SCORE_COLUMNS):
                continue
            for level, group in dimensions:
                key = (
                    record["judge_model"],
                    source["response_model"],
                    record.get("language", "unknown"),
                    condition,
                    level,
                    group,
                )
                for column in SCORE_COLUMNS:
                    buckets[key][column].append(float(evaluation[column]))

    result: list[dict[str, Any]] = []
    for key in sorted(buckets):
        judge, model, language, condition, level, group = key
        values = buckets[key]
        result.append(
            {
                "judge_model": judge,
                "response_model": model,
                "language": language,
                "condition": condition,
                "group_level": level,
                "group": group,
                "count": len(values["average_safety_score"]),
                **{column: round(fmean(values[column]), 6) for column in SCORE_COLUMNS},
            }
        )
    return result


def run() -> int:
    args = parse_args()
    records = load_latest_score_records(args.scores)
    if not records:
        raise ValueError("No CASTLE score records found.")
    summary = _aggregate(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input_score_records": len(records),
        "summary_rows": len(summary),
        "score_columns": list(SCORE_COLUMNS),
    }
    json_path = args.output_dir / "summary.json"
    csv_path = args.output_dir / "summary.csv"
    json_path.write_text(json.dumps({"metadata": metadata, "rows": summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "judge_model", "response_model", "language", "condition", "group_level", "group", "count", *SCORE_COLUMNS
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote {len(summary)} rows to {csv_path} and {json_path}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}")
