"""Custom tool for planning-agent: turns its images-per-class reasoning into
validated plan.md + class_budget.json, instead of trusting the LLM to
hand-write correct JSON via write_file and to correctly apply the
cv-dataset-curation skill's floor/imbalance tables from memory every time.

class_budget.json shape (consumed by dataset-agent's
tools/dataset_builder.py::_canonical_class_list, which reads the "classes"
list from a dict and tolerates this file being absent entirely):

    {
      "classes": ["forklift", "pedestrian"],
      "budgets": {"forklift": 500, "pedestrian": 900},
      "variability": {"forklift": "moderate", "pedestrian": "high"},
      "split_ratios": {"train": 0.7, "val": 0.2, "test": 0.1},
      "notes": "..."
    }
"""

from __future__ import annotations

import json
from langchain_core.tools import tool

from tools.workspace_paths import workspace_path

# Mirrors skills/cv-dataset-curation/SKILL.md's "Images-per-class by
# intra-class variability" table - kept here as an actual enforced floor
# (warn, don't silently trust the model's arithmetic) rather than existing
# only as text the model is told to "consult".
_TIER_FLOORS = {
    "very_low": 100,
    "moderate": 300,
    "high": 800,
    "very_high": 1500,
}


def _normalize_tier(tier: str) -> str:
    return tier.strip().lower().replace(" ", "_").replace("-", "_")


@tool
def write_plan(
    use_case: str,
    deployment_target: str,
    classes: list[dict],
    split_ratios: dict | None = None,
    notes: str = "",
) -> str:
    """Write plan.md and class_budget.json from your images-per-class reasoning.

    `classes` is a list of objects, one per target class:
        {"name": str, "variability_tier": one of "very_low"/"moderate"/"high"/
         "very_high", "images_per_class": int, "rationale": str}

    `split_ratios` defaults to {"train": 0.7, "val": 0.2, "test": 0.1} (per the
    cv-dataset-curation skill's default for datasets under ~2k images) if not
    given; must sum to ~1.0.

    This checks each class's images_per_class against the skill's floor for
    its stated variability tier and flags "thin" classes (under 30% of the
    median class's count, per the skill), warning rather than blocking either
    - you may have deliberate reasons to go below a floor, but the checks
    catch the common cases where the reasoning was skipped, not just trust
    the numbers as given. Writes both files and returns a plain-text summary
    (final counts, ratios, and any warnings) suitable to relay as your final
    message.
    """
    if not classes:
        return "Error: classes list is empty - nothing to plan."

    ratios = split_ratios or {"train": 0.7, "val": 0.2, "test": 0.1}
    ratio_sum = sum(ratios.values())
    if not (0.98 <= ratio_sum <= 1.02):
        return f"Error: split_ratios {ratios} sum to {ratio_sum:.3f}, not ~1.0."

    warnings: list[str] = []
    budgets: dict[str, int] = {}
    variability: dict[str, str] = {}
    class_names: list[str] = []

    for entry in classes:
        name = entry.get("name")
        tier_raw = entry.get("variability_tier", "")
        count = entry.get("images_per_class")
        if not name or count is None:
            warnings.append(f"Skipped malformed class entry (missing name/images_per_class): {entry}")
            continue

        tier = _normalize_tier(str(tier_raw))
        floor = _TIER_FLOORS.get(tier)
        if floor is None:
            warnings.append(
                f"'{name}': unrecognized variability_tier {tier_raw!r} - expected one of "
                f"{sorted(_TIER_FLOORS)}; floor check skipped for this class."
            )
        elif count < floor:
            warnings.append(
                f"'{name}': {count} images is below the {tier} floor of {floor} "
                f"(cv-dataset-curation skill) - double-check this is intentional."
            )

        class_names.append(name)
        budgets[name] = int(count)
        variability[name] = tier

    if budgets:
        median = sorted(budgets.values())[len(budgets) // 2]
        for name, count in budgets.items():
            if median > 0 and count < 0.3 * median:
                warnings.append(
                    f"'{name}': {count} images is under 30% of the median class ({median}) - "
                    f"flagged as 'thin' per the cv-dataset-curation skill; route back to "
                    f"sourcing/annotation before training rather than letting the loss "
                    f"function absorb it."
                )

    budget_data = {
        "classes": class_names,
        "budgets": budgets,
        "variability": variability,
        "split_ratios": ratios,
        "notes": notes,
    }

    plan_lines = [
        f"# Dataset Plan\n",
        f"**Use case:** {use_case}\n",
        f"**Deployment target:** {deployment_target}\n",
        "## Per-class image budget\n",
        "| Class | Variability | Images/class | Rationale |",
        "|---|---|---|---|",
    ]
    for entry in classes:
        plan_lines.append(
            f"| {entry.get('name', '?')} | {entry.get('variability_tier', '?')} | "
            f"{entry.get('images_per_class', '?')} | {entry.get('rationale', '')} |"
        )
    plan_lines.append("")
    plan_lines.append(f"**Split ratios:** train={ratios.get('train')}, val={ratios.get('val')}, test={ratios.get('test')}\n")
    if notes:
        plan_lines.append(f"**Notes:** {notes}\n")
    if warnings:
        plan_lines.append("## Warnings\n")
        for w in warnings:
            plan_lines.append(f"- {w}")

    (workspace_path("/workspace/class_budget.json")).write_text(json.dumps(budget_data, indent=2), encoding="utf-8")
    (workspace_path("/workspace/plan.md")).write_text("\n".join(plan_lines) + "\n", encoding="utf-8")

    total = sum(budgets.values())
    summary_lines = [
        f"Wrote plan.md and class_budget.json for {len(class_names)} class(es), {total} images total.",
        f"Budgets: {budgets}",
        f"Split: train={ratios.get('train')}, val={ratios.get('val')}, test={ratios.get('test')}",
    ]
    if warnings:
        summary_lines.append("Warnings:")
        summary_lines.extend(f"- {w}" for w in warnings)
    return "\n".join(summary_lines)
