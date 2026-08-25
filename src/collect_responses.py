#!/usr/bin/env python3
"""Generate paired personalized and non-personalized responses for CASTLE."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from castle_benchmark.client import ChatClient, ClientSettings, redact_secrets
from castle_benchmark.dataset import DatasetError, Scenario, load_scenarios, sha256_file
from castle_benchmark.io import iter_ndjson, safe_filename, write_ndjson_record
from castle_benchmark.prompts import PROMPT_VERSION, response_messages


ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to CASTLE_zh.json or CASTLE_en.json.")
    parser.add_argument("--models", required=True, help="Comma-separated model identifiers to evaluate.")
    parser.add_argument("--language", choices=("auto", "zh", "en"), default="auto")
    parser.add_argument("--output-dir", default=ROOT_DIR / "results" / "responses", type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None, help="Exclusive dataset index.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum scenarios after slicing.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent scenario workers (default: 1).")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL; defaults to OPENAI_BASE_URL.")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip completed model/scenario pairs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned work without API calls.")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _completed_indices(path: Path, model: str) -> set[int]:
    if not path.exists():
        return set()
    completed: set[int] = set()
    for record in iter_ndjson(path):
        if record.get("record_type") != "castle_model_response" or record.get("model") != model:
            continue
        responses = record.get("responses", {})
        if all(responses.get(condition, {}).get("status") == "ok" for condition in ("non_personalized", "personalized")):
            index = record.get("dataset_index")
            if isinstance(index, int):
                completed.add(index)
    return completed


def _generate_condition(
    client: ChatClient,
    scenario: Scenario,
    model: str,
    *,
    personalized: bool,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    condition = "personalized" if personalized else "non_personalized"
    try:
        completion = client.complete(
            model=model,
            messages=response_messages(
                language=scenario.language,
                query=scenario.user_query,
                profile=scenario.user_profile if personalized else None,
            ),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {"status": "ok", **completion}
    except Exception as exc:
        return {"status": "error", "error": redact_secrets(exc), "condition": condition}


def _generate_record(
    client: ChatClient,
    scenario: Scenario,
    model: str,
    *,
    dataset_name: str,
    dataset_sha256: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    responses = {
        "non_personalized": _generate_condition(
            client, scenario, model, personalized=False, temperature=temperature, max_tokens=max_tokens
        ),
        "personalized": _generate_condition(
            client, scenario, model, personalized=True, temperature=temperature, max_tokens=max_tokens
        ),
    }
    return {
        "schema_version": "1.0.0",
        "record_type": "castle_model_response",
        "created_at": _utc_now(),
        "prompt_version": PROMPT_VERSION,
        "dataset": {"filename": dataset_name, "sha256": dataset_sha256},
        "dataset_index": scenario.index,
        "language": scenario.language,
        "model": model,
        "scenario": scenario.as_dict(include_profile=True),
        "responses": responses,
    }


def _models(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise DatasetError("At least one model is required.")
    names = [safe_filename(item) for item in result]
    if len(set(names)) != len(names):
        raise DatasetError("Model names collide after filename sanitization; run them separately.")
    return result


def run() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise DatasetError("`--workers` must be positive.")
    scenarios = load_scenarios(
        args.input,
        language=args.language,
        start_index=args.start_index,
        end_index=args.end_index,
        limit=args.limit,
    )
    models = _models(args.models)
    input_path = Path(args.input)
    total_pairs = len(scenarios) * len(models)
    print(f"Loaded {len(scenarios)} {scenarios[0].language} scenarios; planned model-scenario pairs: {total_pairs}.")
    if args.dry_run:
        for model in models:
            print(f"Would write {args.output_dir / f'{safe_filename(model)}.ndjson'}")
        return 0

    client = ChatClient(
        ClientSettings(
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        )
    )
    dataset_sha256 = sha256_file(input_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    completed = {
        model: _completed_indices(args.output_dir / f"{safe_filename(model)}.ndjson", model)
        if not args.no_resume
        else set()
        for model in models
    }
    tasks = [
        (model, scenario)
        for model in models
        for scenario in scenarios
        if scenario.index not in completed[model]
    ]
    print(f"Skipping {total_pairs - len(tasks)} completed pairs; processing {len(tasks)} pairs.")

    def make_record(model: str, scenario: Scenario) -> tuple[str, dict[str, Any]]:
        return model, _generate_record(
            client,
            scenario,
            model,
            dataset_name=input_path.name,
            dataset_sha256=dataset_sha256,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    succeeded = failed = 0
    if args.workers == 1:
        iterator = (make_record(model, scenario) for model, scenario in tasks)
        for position, (model, record) in enumerate(iterator, start=1):
            write_ndjson_record(args.output_dir / f"{safe_filename(model)}.ndjson", record)
            if all(item.get("status") == "ok" for item in record["responses"].values()):
                succeeded += 1
            else:
                failed += 1
            if position % 100 == 0 or position == len(tasks):
                print(f"Processed {position}/{len(tasks)} pairs.")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(make_record, model, scenario) for model, scenario in tasks]
            for position, future in enumerate(as_completed(futures), start=1):
                model, record = future.result()
                write_ndjson_record(args.output_dir / f"{safe_filename(model)}.ndjson", record)
                if all(item.get("status") == "ok" for item in record["responses"].values()):
                    succeeded += 1
                else:
                    failed += 1
                if position % 100 == 0 or position == len(tasks):
                    print(f"Processed {position}/{len(tasks)} pairs.")

    print(f"Finished: {succeeded} complete pairs, {failed} pairs with recorded errors.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (DatasetError, FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Error: {redact_secrets(exc)}")
