#!/usr/bin/env python3
"""Generate Appendix A-aligned CASTLE student profiles from existing seed profiles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from castle_benchmark.client import ChatClient, ClientSettings, redact_secrets
from castle_benchmark.construction import (
    TAXONOMY,
    load_json_records,
    make_profile_record,
    parse_json_array,
    scenario_values,
    write_json,
)
from castle_benchmark.generation_prompts import create_profile_generation_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-file", required=True, help="Existing CASTLE-format JSON used as profile seeds.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON containing scenario type, subtype, and profile.")
    parser.add_argument("--models", required=True, help="Comma-separated OpenAI-compatible model identifiers.")
    parser.add_argument("--language", choices=("zh", "en"), required=True)
    parser.add_argument("--profiles-per-scenario", type=int, default=1)
    parser.add_argument("--examples-per-scenario", type=int, default=3)
    parser.add_argument("--limit-scenarios", type=int, default=None, help="Useful for a small pilot run.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)
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


def seeds_by_scenario(records: list[dict[str, Any]], language: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        scenario_type, scenario_subtype, profile = scenario_values(record, language)  # type: ignore[arg-type]
        result.setdefault((scenario_type, scenario_subtype), []).append(profile)
    return result


def profiles_from_completion(text: str) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for item in parse_json_array(text):
        if not isinstance(item, dict):
            continue
        profile = item.get("Profile", item.get("画像", item))
        if isinstance(profile, dict) and "Stable_Attributes" in profile or isinstance(profile, dict) and "稳定属性" in profile:
            profiles.append(profile)
    if not profiles:
        raise ValueError("The model output did not contain valid profile objects.")
    return profiles


def run() -> int:
    args = parse_args()
    if args.profiles_per_scenario <= 0 or args.examples_per_scenario <= 0:
        raise ValueError("Profile and example counts must be positive.")
    language, seed_records = load_json_records(args.seed_file, args.language)
    models = models_from_argument(args.models)
    entries = TAXONOMY[: args.limit_scenarios] if args.limit_scenarios else TAXONOMY
    seed_index = seeds_by_scenario(seed_records, language)
    missing = [entry.labels(language) for entry in entries if entry.labels(language) not in seed_index]
    if missing:
        preview = ", ".join(f"{scenario_type}/{subtype}" for scenario_type, subtype in missing[:3])
        raise ValueError(f"Seed file has no matching profiles for {len(missing)} requested scenarios, e.g. {preview}")
    print(f"Using {len(seed_records)} seed records to generate {len(entries) * args.profiles_per_scenario} profiles.")
    if args.dry_run:
        print(f"Would write generated {language} profiles to {args.output}.")
        return 0

    client = ChatClient(
        ClientSettings(
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        )
    )
    generated_records: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        scenario_type, scenario_subtype = entry.labels(language)
        examples = seed_index[(scenario_type, scenario_subtype)][: args.examples_per_scenario]
        prompt = create_profile_generation_prompt(
            scenario_type,
            scenario_subtype,
            examples,
            args.profiles_per_scenario,
            language,
        )
        completion = client.complete(
            model=models[index % len(models)],
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        profiles = profiles_from_completion(completion["text"])
        if len(profiles) < args.profiles_per_scenario:
            raise ValueError(f"{scenario_subtype}: expected {args.profiles_per_scenario} profiles, received {len(profiles)}.")
        generated_records.extend(
            make_profile_record(entry, language, profile) for profile in profiles[: args.profiles_per_scenario]
        )
        print(f"Generated {args.profiles_per_scenario} profiles for {scenario_subtype}.")
    write_json(args.output, generated_records)
    print(f"Wrote {len(generated_records)} profiles to {args.output}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Error: {redact_secrets(exc)}")
