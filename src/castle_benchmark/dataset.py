"""Dataset loading, normalization, and validation for CASTLE."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal


Language = Literal["zh", "en"]


class DatasetError(ValueError):
    """Raised when a CASTLE dataset file does not match the expected schema."""


@dataclass(frozen=True)
class Scenario:
    """Language-neutral representation of one CASTLE benchmark scenario."""

    index: int
    language: Language
    scenario_type: str
    scenario_subtype: str
    user_query: str
    user_profile: dict[str, Any]

    def as_dict(self, *, include_profile: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "scenario_type": self.scenario_type,
            "scenario_subtype": self.scenario_subtype,
            "user_query": self.user_query,
        }
        if include_profile:
            result["user_profile"] = self.user_profile
        return result


_ZH_FIELDS = {
    "scenario_type": "场景类型",
    "scenario_subtype": "场景子类型",
    "user_query": "用户查询",
    "user_profile": "用户画像",
}
_EN_FIELDS = {
    "scenario_type": "Scenario_Type",
    "scenario_subtype": "Scenario_Subtype",
    "user_query": "User_Query",
    "user_profile": "User_Profile",
}


def infer_language(record: dict[str, Any], path: str | Path | None = None) -> Language:
    """Infer the data language from canonical field names, then from its filename."""
    if "Scenario_Type" in record:
        return "en"
    if "场景类型" in record:
        return "zh"
    if path is not None and "_en" in Path(path).stem.lower():
        return "en"
    raise DatasetError("Cannot infer language: expected English or Chinese CASTLE field names.")


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(record: dict[str, Any], field: str, index: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"Record {index}: missing or empty `{field}`.")
    return value.strip()


def normalize_record(record: dict[str, Any], index: int, language: Language) -> Scenario:
    """Validate and convert one raw Chinese or English record to :class:`Scenario`."""
    if not isinstance(record, dict):
        raise DatasetError(f"Record {index}: expected an object, got {type(record).__name__}.")
    fields = _ZH_FIELDS if language == "zh" else _EN_FIELDS
    profile = record.get(fields["user_profile"])
    if not isinstance(profile, dict) or not profile:
        raise DatasetError(f"Record {index}: missing or invalid `{fields['user_profile']}`.")
    return Scenario(
        index=index,
        language=language,
        scenario_type=_required_text(record, fields["scenario_type"], index),
        scenario_subtype=_required_text(record, fields["scenario_subtype"], index),
        user_query=_required_text(record, fields["user_query"], index),
        user_profile=profile,
    )


def load_scenarios(
    path: str | Path,
    *,
    language: Language | Literal["auto"] = "auto",
    start_index: int = 0,
    end_index: int | None = None,
    limit: int | None = None,
) -> list[Scenario]:
    """Load and validate a range of scenarios from a CASTLE JSON array."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {input_path}")
    if start_index < 0:
        raise DatasetError("`start_index` must be non-negative.")
    if limit is not None and limit <= 0:
        raise DatasetError("`limit` must be positive when provided.")

    with input_path.open("r", encoding="utf-8") as handle:
        raw_records = json.load(handle)
    if not isinstance(raw_records, list) or not raw_records:
        raise DatasetError("A CASTLE dataset must be a non-empty JSON array.")

    detected_language = infer_language(raw_records[0], input_path)
    if language != "auto" and language != detected_language:
        raise DatasetError(
            f"Language mismatch: --language={language}, but data appears to be {detected_language}."
        )

    stop = len(raw_records) if end_index is None else min(end_index, len(raw_records))
    if stop < start_index:
        raise DatasetError("`end_index` must be greater than or equal to `start_index`.")
    selected = raw_records[start_index:stop]
    if limit is not None:
        selected = selected[:limit]

    return [
        normalize_record(record, start_index + offset, detected_language)
        for offset, record in enumerate(selected)
    ]


def summarize_scenarios(scenarios: Iterable[Scenario]) -> dict[str, Any]:
    """Build portable schema and taxonomy statistics for a scenario collection."""
    scenario_list = list(scenarios)
    language_counts = Counter(item.language for item in scenario_list)
    type_counts = Counter(item.scenario_type for item in scenario_list)
    subtype_counts = Counter(item.scenario_subtype for item in scenario_list)
    return {
        "records": len(scenario_list),
        "language_counts": dict(sorted(language_counts.items())),
        "scenario_type_counts": dict(sorted(type_counts.items())),
        "scenario_subtype_counts": dict(sorted(subtype_counts.items())),
    }
