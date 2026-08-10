"""
Preprocessing stage for the Medical Guiding System.

Responsibility of this module: take loaded Document objects and normalize
them into a consistent per-condition shape, so the extraction stage never
has to branch on which guideline file it's looking at.

Why this exists: confirmed during document loading that management_plan
is structured differently across files, e.g.

    clinical_guidelines_headache.json:
        management_plan.pharmacological.{acute_abortive, preventive}
        management_plan.non_pharmacological

    all other files:
        management_plan.initial_conservative
        management_plan.specific_advanced
        management_plan.follow_up_monitoring

Both get normalized into one shape: {"conservative": [...], "advanced": [...],
"follow_up": [...]}. Anything in an unrecognized shape is not dropped — its
list-valued leaves are flattened into "advanced" and a warning is logged, so
a future guideline with yet another management_plan shape degrades safely
instead of crashing the pipeline.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.document_loader.schema import Document

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedCondition:
    """One normalized condition, ready for the extraction stage."""

    condition_name: str
    key_features: list[str] = field(default_factory=list)
    investigations: list[str] = field(default_factory=list)
    differential_red_flags: list[str] = field(default_factory=list)
    management_conservative: list[str] = field(default_factory=list)
    management_advanced: list[str] = field(default_factory=list)
    management_follow_up: list[str] = field(default_factory=list)
    raw_condition: dict[str, Any] = field(default_factory=dict)


def clean_text(value: str) -> str:
    """Trim and collapse internal whitespace. Leaves clinical punctuation intact."""
    if not isinstance(value, str):
        return value
    return re.sub(r"\s+", " ", value).strip()


def clean_list(values: list) -> list[str]:
    if not isinstance(values, list):
        return []
    return [clean_text(v) for v in values if isinstance(v, str) and clean_text(v)]


def _flatten_list_leaves(node: Any, out: list[str]) -> None:
    """Recursively collect every string found in list-valued leaves of a dict/list."""
    if isinstance(node, list):
        for item in node:
            if isinstance(item, str):
                out.append(item)
            else:
                _flatten_list_leaves(item, out)
    elif isinstance(node, dict):
        for value in node.values():
            _flatten_list_leaves(value, out)


def normalize_management_plan(condition: dict, context_name: str) -> dict:
    """
    Normalize a condition's management_plan into:
        {"conservative": [...], "advanced": [...], "follow_up": [...]}
    regardless of which of the known source shapes it arrived in.
    """
    plan = condition.get("management_plan", {})
    if not isinstance(plan, dict):
        return {"conservative": [], "advanced": [], "follow_up": []}

    # Shape A: initial_conservative / specific_advanced / follow_up_monitoring
    if "initial_conservative" in plan or "specific_advanced" in plan:
        return {
            "conservative": clean_list(plan.get("initial_conservative", [])),
            "advanced": clean_list(plan.get("specific_advanced", [])),
            "follow_up": clean_list(plan.get("follow_up_monitoring", [])),
        }

    # Shape B: pharmacological{acute_abortive,preventive} / non_pharmacological
    if "pharmacological" in plan or "non_pharmacological" in plan:
        pharma = plan.get("pharmacological", {})
        advanced: list[str] = []
        if isinstance(pharma, dict):
            advanced.extend(clean_list(pharma.get("acute_abortive", [])))
            advanced.extend(clean_list(pharma.get("preventive", [])))
        elif isinstance(pharma, list):
            advanced.extend(clean_list(pharma))
        return {
            "conservative": clean_list(plan.get("non_pharmacological", [])),
            "advanced": advanced,
            "follow_up": [],  # this shape doesn't carry a distinct follow-up bucket
        }

    # Unknown shape: don't drop the content, flatten it and warn so the gap is visible.
    logger.warning(
        "%s: condition %r has an unrecognized management_plan shape (keys=%s) "
        "— flattening into 'advanced' instead of structured buckets",
        context_name, condition.get("condition_name"), list(plan.keys()),
    )
    flattened: list[str] = []
    _flatten_list_leaves(plan, flattened)
    return {"conservative": [], "advanced": clean_list(flattened), "follow_up": []}


def preprocess_condition(condition: dict, context_name: str) -> PreprocessedCondition:
    presentation = condition.get("clinical_presentation", {}) or {}
    diagnostics = condition.get("diagnostic_approach", {}) or {}
    management = normalize_management_plan(condition, context_name)

    return PreprocessedCondition(
        condition_name=clean_text(condition.get("condition_name", "Unnamed condition")),
        key_features=clean_list(presentation.get("key_features", [])),
        investigations=clean_list(diagnostics.get("investigations", [])),
        differential_red_flags=clean_list(diagnostics.get("differential_red_flags", [])),
        management_conservative=management["conservative"],
        management_advanced=management["advanced"],
        management_follow_up=management["follow_up"],
        raw_condition=condition,
    )


def preprocess_document(document: Document) -> list[PreprocessedCondition]:
    """
    Normalize every condition in a loaded Document.
    Returns one PreprocessedCondition per entry in conditions_registry.
    """
    return [
        preprocess_condition(condition, context_name=document.doc_id)
        for condition in document.conditions_registry
    ]
