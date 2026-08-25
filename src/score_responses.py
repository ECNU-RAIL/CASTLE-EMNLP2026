#!/usr/bin/env python3
"""Score CASTLE response artifacts with the three-dimension LLM-as-a-judge protocol."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from castle_benchmark.client import ChatClient, ClientSettings, redact_secrets
from castle_benchmark.io import collect_ndjson_paths, iter_ndjson, safe_filename, write_ndjson_record
from castle_benchmark.prompts import PROMPT_VERSION, evaluation_messages
from castle_benchmark.scoring import ScoreParseError, parse_judge_output


ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", required=True, help="A response .ndjson file or directory from collect_responses.py.")
    parser.add_argument("--judge", required=True, help="Model identifier for the LLM judge.")
    parser.add_argument("--output-dir", default=ROOT_DIR / "results" / "scores", type=Path)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent response-record workers (default: 1).")
    parser.add_argument("--limit", type=int, default=None, help="Maximum response records after de-duplication.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL; defaults to OPENAI_BASE_URL.")
    parser.add_argument("--store-raw-judge-output", action="store_true", help="Keep raw judge text in local result artifacts.")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip fully scored records.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned work without API calls.")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _response_key(record: dict[str, Any]) -> tuple[str, int]:
    model = record.get("model")
    index = record.get("dataset_index")
    if not isinstance(model, str) or not isinstance(index, int):
        raise ValueError("Response record requires string `model` and integer `dataset_index`.")
    return model, index


def load_latest_response_records(path: str | Path) -> list[dict[str, Any]]:
    """Read one or more response files; later append-only retry records supersede earlier ones."""
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for input_path in collect_ndjson_paths(path):
        for record in iter_ndjson(input_path):
            if record.get("record_type") != "castle_model_response":
                continue
            latest[_response_key(record)] = record
    return [latest[key] for key in sorted(latest, key=lambda item: (item[0], item[1]))]


def _completed_keys(path: Path, judge: str) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, int]] = set()
    for record in iter_ndjson(path):
        if record.get("record_type") != "castle_response_score" or record.get("judge_model") != judge:
            continue
        evaluations = record.get("evaluations", {})
        if all(evaluations.get(condition, {}).get("status") == "ok" for condition in ("non_personalized", "personalized")):
            source = record.get("source", {})
            model, index = source.get("response_model"), record.get("dataset_index")
            if isinstance(model, str) and isinstance(index, int):
                completed.add((model, index))
    return completed


def _score_condition(
    client: ChatClient,
    *,
    judge: str,
    language: str,
    query: str,
    profile: dict[str, Any],
    response: dict[str, Any],
    temperature: float,
    max_tokens: int,
    store_raw_output: bool,
) -> dict[str, Any]:
    if response.get("status") != "ok" or not isinstance(response.get("text"), str):
        return {"status": "skipped", "reason": "The source response was not generated successfully."}
    try:
        completion = client.complete(
            model=judge,
            messages=evaluation_messages(
                language=language,  # type: ignore[arg-type]
                response=response["text"],
                query=query,
                profile=profile,
            ),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = parse_judge_output(completion["text"])
        result: dict[str, Any] = {
            "status": "ok",
            **parsed,
            "finish_reason": completion["finish_reason"],
            "usage": completion["usage"],
        }
        if store_raw_output:
            result["raw_judge_output"] = completion["text"]
        return result
    except (RuntimeError, ScoreParseError, ValueError) as exc:
        return {"status": "error", "error": redact_secrets(exc)}


def _score_record(
    client: ChatClient,
    response_record: dict[str, Any],
    *,
    judge: str,
    temperature: float,
    max_tokens: int,
    store_raw_output: bool,
) -> dict[str, Any]:
    model, dataset_index = _response_key(response_record)
    scenario = response_record.get("scenario")
    if not isinstance(scenario, dict):
        raise ValueError(f"Response record {model}/{dataset_index} has no scenario object.")
    language = response_record.get("language")
    query, profile = scenario.get("user_query"), scenario.get("user_profile")
    if language not in {"zh", "en"} or not isinstance(query, str) or not isinstance(profile, dict):
        raise ValueError(f"Response record {model}/{dataset_index} has incomplete scenario data.")

    evaluations = {
        condition: _score_condition(
            client,
            judge=judge,
            language=language,
            query=query,
            profile=profile,
            response=response_record.get("responses", {}).get(condition, {}),
            temperature=temperature,
            max_tokens=max_tokens,
            store_raw_output=store_raw_output,
        )
        for condition in ("non_personalized", "personalized")
    }
    return {
        "schema_version": "1.0.0",
        "record_type": "castle_response_score",
        "created_at": _utc_now(),
        "prompt_version": PROMPT_VERSION,
        "judge_model": judge,
        "dataset_index": dataset_index,
        "language": language,
        "source": {
            "response_model": model,
            "response_dataset": response_record.get("dataset", {}),
        },
        "scenario": scenario,
        "evaluations": evaluations,
    }


def run() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("`--workers` must be positive.")
    records = load_latest_response_records(args.responses)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("`--limit` must be positive when provided.")
        records = records[: args.limit]
    if not records:
        raise ValueError("No CASTLE response records found.")

    output_path = args.output_dir / f"{safe_filename(args.judge)}.ndjson"
    completed = _completed_keys(output_path, args.judge) if not args.no_resume else set()
    pending = [record for record in records if _response_key(record) not in completed]
    print(f"Loaded {len(records)} latest response records; skipping {len(records) - len(pending)} completed records.")
    if args.dry_run:
        print(f"Would score {len(pending)} records with `{args.judge}` into {output_path}.")
        return 0

    client = ChatClient(
        ClientSettings(
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        )
    )

    def score(record: dict[str, Any]) -> dict[str, Any]:
        return _score_record(
            client,
            record,
            judge=args.judge,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            store_raw_output=args.store_raw_judge_output,
        )

    complete = partial = 0
    if args.workers == 1:
        iterator = (score(record) for record in pending)
        for position, result in enumerate(iterator, start=1):
            write_ndjson_record(output_path, result)
            if all(item.get("status") == "ok" for item in result["evaluations"].values()):
                complete += 1
            else:
                partial += 1
            if position % 100 == 0 or position == len(pending):
                print(f"Scored {position}/{len(pending)} response records.")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(score, record) for record in pending]
            for position, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                write_ndjson_record(output_path, result)
                if all(item.get("status") == "ok" for item in result["evaluations"].values()):
                    complete += 1
                else:
                    partial += 1
                if position % 100 == 0 or position == len(pending):
                    print(f"Scored {position}/{len(pending)} response records.")
    print(f"Finished: {complete} complete score records, {partial} partial/error records.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Error: {redact_secrets(exc)}")
