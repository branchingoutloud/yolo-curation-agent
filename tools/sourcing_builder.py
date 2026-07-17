"""Custom tool for sourcing-agent: code-enforces the "read sources.json first,
then append/update - never overwrite" contract instead of relying on the
model to remember and correctly hand-write JSON via write_file every call.
This is the exact contract dataset-agent (and the orchestrator's routing
decisions) depend on - see tools/dataset_builder.py and subagents/dataset.py.

Required shape per source entry (matches what dataset-agent expects):
    {
      "source": "roboflow_universe" | "kaggle" | ...,
      "dataset_id": str,
      "url": str,
      "classes_covered": [str, ...],
      "image_count": int,
      "annotation_format": "YOLO" | "Pascal VOC" | ...,
      "license": str,
      "annotation_coverage": {class_name: int | null, ...},
      "quality_notes": str,
      "status": "available" | "needs_annotation" | ...
    }
"""

from __future__ import annotations

import json
from langchain_core.tools import tool

from tools.workspace_paths import workspace_path

_REQUIRED_FIELDS = {"source", "dataset_id", "status"}


@tool
def append_sources(new_sources: list[dict], sources_json_path: str = "/workspace/sources.json") -> str:
    """Merge new source candidates into sources.json - reads the existing
    file first (if any) and updates-or-appends by (source, dataset_id) key,
    never overwriting entries you didn't just find. Each entry needs at least
    source/dataset_id/status; see this module's docstring for the full shape
    (classes_covered, image_count, annotation_format, license,
    annotation_coverage, quality_notes) dataset-agent expects downstream.

    Returns a plain-text summary (N added, M updated, current class coverage)
    suitable to relay as your final message, in the same spirit as: "Found 3
    sources covering 'car' (1,850 images); 'bus' unlabeled - 300 raw images,
    0 annotated. Recommend routing 'bus' to annotation."
    """
    try:
        path = workspace_path(sources_json_path)
    except ValueError as exc:
        return f"Error: {exc}. Pass a path starting with /workspace/ instead (or omit it - the default is correct)."

    existing: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                return f"Error: {sources_json_path} exists but isn't a JSON array - refusing to overwrite it."
        except json.JSONDecodeError as exc:
            return f"Error: {sources_json_path} exists but isn't valid JSON ({exc}) - refusing to overwrite it."

    by_key = {(e.get("source"), e.get("dataset_id")): i for i, e in enumerate(existing)}

    added, updated, rejected = 0, 0, []
    for entry in new_sources:
        missing = _REQUIRED_FIELDS - entry.keys()
        if missing:
            rejected.append(f"{entry.get('dataset_id', '?')}: missing required field(s) {sorted(missing)}")
            continue

        key = (entry.get("source"), entry.get("dataset_id"))
        if key in by_key:
            existing[by_key[key]] = entry
            updated += 1
        else:
            existing.append(entry)
            by_key[key] = len(existing) - 1
            added += 1

    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    coverage: dict[str, int] = {}
    for entry in existing:
        for cls in entry.get("classes_covered", []):
            coverage[cls] = coverage.get(cls, 0) + int(entry.get("image_count") or 0)

    summary_lines = [
        f"sources.json: {added} added, {updated} updated, {len(existing)} total.",
        f"Class coverage (summed image_count across all sources, available or not): {coverage}",
    ]
    if rejected:
        summary_lines.append("Rejected (not written):")
        summary_lines.extend(f"- {r}" for r in rejected)
    return "\n".join(summary_lines)
