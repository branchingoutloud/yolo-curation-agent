"""The sources.json contract between sourcing-agent and everything downstream.

sourcing-agent is prompt-driven, so nothing here runs inside the agent loop —
these are pure helpers used by the isolation harness and tests to verify what
the agent actually wrote. The schema mirrors sourcing-subagent-deep-dive.md;
the `status` field is the orchestrator's routing signal (a source marked
"available" when it has no labels would silently feed unannotated data into
training), which is why validation is strict about it.
"""

from typing import Any

REQUIRED_FIELDS: tuple[str, ...] = (
    "source",
    "dataset_id",
    "url",
    "classes_covered",
    "image_count",
    "annotation_format",
    "license",
    "annotation_coverage",
    "status",
)

# available: annotated and directly downloadable/forkable
# needs_annotation: raw images obtainable, but no (usable) labels
# unannotated_collection: web pointer only — images not directly downloadable
VALID_STATUSES: frozenset[str] = frozenset(
    {"available", "needs_annotation", "unannotated_collection"}
)


def validate_sources(sources: Any) -> list[str]:
    """Validate a parsed sources.json document. Returns a list of error
    strings, empty when the document conforms to the contract.
    """
    if not isinstance(sources, list):
        return ["sources.json must be a JSON array of source entries"]

    errors: list[str] = []
    for i, entry in enumerate(sources):
        prefix = f"[{i}]"
        if not isinstance(entry, dict):
            errors.append(f"{prefix} entry must be an object, got {type(entry).__name__}")
            continue

        missing = [f for f in REQUIRED_FIELDS if f not in entry]
        errors.extend(f"{prefix} missing required field '{f}'" for f in missing)

        status = entry.get("status")
        if "status" not in missing and status not in VALID_STATUSES:
            errors.append(
                f"{prefix} invalid status {status!r} — must be one of "
                f"{sorted(VALID_STATUSES)}"
            )

        image_count = entry.get("image_count")
        if "image_count" not in missing and (
            not isinstance(image_count, int) or image_count < 0
        ):
            errors.append(f"{prefix} image_count must be a non-negative integer")

        classes_covered = entry.get("classes_covered")
        coverage = entry.get("annotation_coverage")
        if isinstance(classes_covered, list) and isinstance(coverage, dict):
            stray = sorted(set(coverage) - set(classes_covered))
            errors.extend(
                f"{prefix} annotation_coverage has class '{c}' not listed in "
                "classes_covered"
                for c in stray
            )

    return errors


def find_duplicate_dataset_ids(sources: list[dict]) -> list[str]:
    """Dataset ids appearing more than once — the append-not-overwrite rule
    means re-runs must update existing entries, not duplicate them.
    """
    seen: set[str] = set()
    duplicates: list[str] = []
    for entry in sources:
        dataset_id = entry.get("dataset_id")
        if dataset_id in seen and dataset_id not in duplicates:
            duplicates.append(dataset_id)
        elif dataset_id is not None:
            seen.add(dataset_id)
    return duplicates


def coverage_summary(sources: list[dict], class_budget: dict) -> dict[str, dict]:
    """Per-class coverage vs the class_budget.json targets.

    annotated: summed annotation_coverage from "available" sources only.
    raw: image_count from needs_annotation/unannotated_collection sources
    covering the class — obtainable but useless for training until labeled.
    """
    summary: dict[str, dict] = {}
    for cls in class_budget.get("classes", []):
        name, target = cls["name"], cls["target_images"]
        annotated = sum(
            entry.get("annotation_coverage", {}).get(name, 0)
            for entry in sources
            if entry.get("status") == "available"
        )
        raw = sum(
            entry.get("image_count", 0)
            for entry in sources
            if entry.get("status") in ("needs_annotation", "unannotated_collection")
            and name in entry.get("classes_covered", [])
        )
        summary[name] = {
            "target": target,
            "annotated": annotated,
            "raw": raw,
            "gap": max(0, target - annotated),
            "met": annotated >= target,
        }
    return summary
