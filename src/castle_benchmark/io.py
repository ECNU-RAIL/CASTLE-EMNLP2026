"""Small, dependency-free helpers for resumable JSON Lines artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator


def safe_filename(value: str) -> str:
    """Convert a model identifier into a filesystem-safe, readable basename."""
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def write_ndjson_record(path: str | Path, record: dict[str, Any]) -> None:
    """Append one record and force it to disk for interruption-safe processing."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def iter_ndjson(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records from a JSON Lines file and identify malformed line numbers."""
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {input_path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {input_path}:{line_number}")
            yield payload


def collect_ndjson_paths(path: str | Path) -> list[Path]:
    """Accept one NDJSON file or a directory of NDJSON files."""
    input_path = Path(path)
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        paths = sorted(item for item in input_path.glob("*.ndjson") if item.is_file())
        if paths:
            return paths
    raise FileNotFoundError(f"No .ndjson file found at: {input_path}")
