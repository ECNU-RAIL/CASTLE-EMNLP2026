#!/usr/bin/env python3
"""Generate CASTLE queries from profile records using the original dual-prompt protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from castle_benchmark.client import ChatClient, ClientSettings, redact_secrets
from castle_benchmark.construction import (
    TAXONOMY,
    clean_queries,
    field_names,
    load_json_records,
    make_query_record,
    scenario_values,
    write_json,
)
from castle_benchmark.generation_prompts import create_query_generation_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Profile-only CASTLE JSON from generate_profiles.py.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON with one record per generated query.")
    parser.add_argument("--models", required=True, help="Comma-separated OpenAI-compatible model identifiers.")
    parser.add_argument("--language", choices=("zh", "en"), required=True)
    parser.add_argument("--queries-per-profile", type=int, default=5)
    parser.add_argument("--prompt-variant", choices=("rotate", "short", "rich"), default="rotate")
    parser.add_argument("--limit", type=int, default=None, help="Useful for a small pilot run.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def models_from_argument(value: str) -> list[str]:
    models = [model.strip() for model in value.split(",") if model.strip()]
    if not models:
        raise ValueError("At least one model is required.")
    return models


def choose_model_and_variant(index: int, models: list[str], requested_variant: str) -> tuple[str, str]:
    model = models[index % len(models)]
    if requested_variant != "rotate":
        return model, requested_variant
    return model, ("short" if (index // len(models)) % 2 == 0 else "rich")


def lookup_entry(scenario_type: str, scenario_subtype: str, language: str):
    for entry in TAXONOMY:
        if entry.labels(language) == (scenario_type, scenario_subtype):
            return entry
    raise ValueError(f"Unknown CASTLE scenario: {scenario_type}/{scenario_subtype}")


def run() -> int:
    args = parse_args()
    if args.queries_per_profile <= 0:
        raise ValueError("`--queries-per-profile` must be positive.")
    language, records = load_json_records(args.input, args.language)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("`--limit` must be positive when provided.")
        records = records[: args.limit]
    models = models_from_argument(args.models)
    fields = field_names(language)
    if any(fields["query"] in record for record in records):
        raise ValueError("Input already contains queries; use a profile-only file to avoid accidental duplication.")
    print(f"Using {len(records)} {language} profiles to generate {len(records) * args.queries_per_profile} queries.")
    if args.dry_run:
        print(f"Would write generated {language} scenarios to {args.output}.")
        return 0

    client = ChatClient(
        ClientSettings(
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        )
    )
    output_records = []
    for index, record in enumerate(records):
        scenario_type, scenario_subtype, profile = scenario_values(record, language)
        entry = lookup_entry(scenario_type, scenario_subtype, language)
        model, variant = choose_model_and_variant(index, models, args.prompt_variant)
        prompt = create_query_generation_prompt(
            scenario_type,
            scenario_subtype,
            profile,
            num_queries=args.queries_per_profile,
            variant=variant,
            language=language,
        )
        completion = client.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        queries = clean_queries(completion["text"], args.queries_per_profile)
        output_records.extend(make_query_record(entry, language, profile, query) for query in queries)
        print(f"Generated {len(queries)} queries for profile {index + 1}/{len(records)}.")
    write_json(args.output, output_records)
    print(f"Wrote {len(output_records)} scenarios to {args.output}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Error: {redact_secrets(exc)}")
