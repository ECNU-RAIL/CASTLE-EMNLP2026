"""Shared schema, taxonomy, and parsing helpers for CASTLE data construction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


Language = Literal["zh", "en"]


@dataclass(frozen=True)
class TaxonomyEntry:
    zh_type: str
    zh_subtype: str
    en_type: str
    en_subtype: str

    def labels(self, language: Language) -> tuple[str, str]:
        if language == "zh":
            return self.zh_type, self.zh_subtype
        return self.en_type, self.en_subtype


TAXONOMY: tuple[TaxonomyEntry, ...] = (
    TaxonomyEntry("心理与情感健康", "学业压力过载", "Psychological and Emotional Health", "Academic Pressure Overload"),
    TaxonomyEntry("心理与情感健康", "生涯选择困境", "Psychological and Emotional Health", "Career Choice Dilemma"),
    TaxonomyEntry("心理与情感健康", "人际关系问题", "Psychological and Emotional Health", "Interpersonal Relationship Issues"),
    TaxonomyEntry("心理与情感健康", "虚拟情感依赖与社交退化", "Psychological and Emotional Health", "Virtual Emotional Dependence and Social Withdrawal"),
    TaxonomyEntry("心理与情感健康", "AI 诱导的自我认知偏差", "Psychological and Emotional Health", "AI-Induced Self-Cognitive Bias"),
    TaxonomyEntry("学术诚信与能力", "学术不端", "Academic Integrity and Competence", "Academic Misconduct"),
    TaxonomyEntry("学术诚信与能力", "规避学习路径", "Academic Integrity and Competence", "Avoiding Learning Path"),
    TaxonomyEntry("学术诚信与能力", "评估质疑", "Academic Integrity and Competence", "Assessment Questioning"),
    TaxonomyEntry("内容与信息偏差", "刻板印象/歧视", "Content and Information Bias", "Stereotype / Discrimination"),
    TaxonomyEntry("内容与信息偏差", "模型幻觉风险", "Content and Information Bias", "Model Hallucination Risks"),
    TaxonomyEntry("内容与信息偏差", "不适宜内容", "Content and Information Bias", "Inappropriate Content"),
    TaxonomyEntry("内容与信息偏差", "知识局限", "Content and Information Bias", "Knowledge Limitations"),
    TaxonomyEntry("学习依赖与认知", "过度寻求简化", "Learning Dependence and Cognition", "Excessive Seeking Simplification"),
    TaxonomyEntry("学习依赖与认知", "丧失独立判断", "Learning Dependence and Cognition", "Loss of Independent Judgment"),
    TaxonomyEntry("学习依赖与认知", "思维固化与创新抑制", "Learning Dependence and Cognition", "Cognitive Rigidity and Innovation Suppression"),
)


def field_names(language: Language) -> dict[str, str]:
    if language == "zh":
        return {"type": "场景类型", "subtype": "场景子类型", "profile": "用户画像", "query": "用户查询"}
    return {"type": "Scenario_Type", "subtype": "Scenario_Subtype", "profile": "User_Profile", "query": "User_Query"}


def infer_language(records: list[Any]) -> Language:
    if not records or not isinstance(records[0], dict):
        raise ValueError("Expected a non-empty JSON array of objects.")
    if "场景类型" in records[0]:
        return "zh"
    if "Scenario_Type" in records[0]:
        return "en"
    raise ValueError("Could not infer dataset language from scenario field names.")


def load_json_records(path: str | Path, language: Language | Literal["auto"] = "auto") -> tuple[Language, list[dict[str, Any]]]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty JSON array: {input_path}")
    detected = infer_language(records)
    if language != "auto" and language != detected:
        raise ValueError(f"Requested language `{language}` does not match `{detected}` input data.")
    return detected, records


def scenario_values(record: dict[str, Any], language: Language) -> tuple[str, str, dict[str, Any]]:
    fields = field_names(language)
    scenario_type = record.get(fields["type"])
    scenario_subtype = record.get(fields["subtype"])
    profile = record.get(fields["profile"])
    if not isinstance(scenario_type, str) or not isinstance(scenario_subtype, str) or not isinstance(profile, dict):
        raise ValueError("Every construction record needs scenario type, subtype, and a profile object.")
    return scenario_type, scenario_subtype, profile


def make_profile_record(entry: TaxonomyEntry, language: Language, profile: dict[str, Any]) -> dict[str, Any]:
    fields = field_names(language)
    scenario_type, scenario_subtype = entry.labels(language)
    return {fields["type"]: scenario_type, fields["subtype"]: scenario_subtype, fields["profile"]: profile}


def make_query_record(entry: TaxonomyEntry, language: Language, profile: dict[str, Any], query: str) -> dict[str, Any]:
    record = make_profile_record(entry, language, profile)
    record[field_names(language)["query"]] = query
    return record


def parse_json_array(text: str) -> list[Any]:
    """Extract the first JSON array from a completion, tolerating fenced output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3].rstrip()
    decoder = json.JSONDecoder()
    for start, character in enumerate(cleaned):
        if character != "[":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return value
    raise ValueError("Model output did not contain a JSON array.")


def clean_queries(text: str, expected_count: int) -> list[str]:
    """Convert the one-query-per-line output protocol into clean query strings."""
    queries: list[str] = []
    for line in text.splitlines():
        value = re.sub(r"^\s*(?:\d+[.)、:-]|[-*•])\s*", "", line).strip().strip('"')
        if value:
            queries.append(value)
    if len(queries) < expected_count:
        raise ValueError(f"Expected {expected_count} queries, received {len(queries)}.")
    return queries[:expected_count]


def write_json(path: str | Path, records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
