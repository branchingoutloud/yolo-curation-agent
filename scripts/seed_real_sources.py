"""Seed a workspace with a REAL sources.json (actual Roboflow Universe
datasets a human found) instead of the synthetic dummy data in
seed_dummy_workspace.py. Deliberately does NOT stage any local files under
sourced/<i>/ - the whole point of this seed is to test whether dataset-agent
can actually discover and use its Roboflow MCP tools to fetch real data
itself, not to hand it pre-downloaded images.

Requires ROBOFLOW_API_KEY set in .env for dataset-agent to have any Roboflow
tools at all; without it, dataset-agent will correctly report that these
sources can't be merged yet (nothing staged, no tools to fetch them) rather
than fabricate anything - that's expected, not a bug, if you haven't set the
key yet.
"""

import json
import sys
from pathlib import Path

REAL_SOURCES = [
    {
        "source": "roboflow_universe",
        "dataset_id": "trainmodel/gender-detection-qiyyg",
        "url": "https://universe.roboflow.com/trainmodel/gender-detection-qiyyg",
        "classes_covered": ["male", "female"],
        "image_count": 2302,
        "annotation_format": "YOLO",
        "license": "unknown",
        "annotation_coverage": {"male": None, "female": None},
        "quality_notes": "pre-trained gender detection model + API available",
        "status": "available",
    },
    {
        "source": "roboflow_universe",
        "dataset_id": "gender-classification-aflfc/gender-detection-and-labelling",
        "url": "https://universe.roboflow.com/gender-classification-aflfc/gender-detection-and-labelling",
        "classes_covered": ["male", "female", "FOCG"],
        "image_count": 551,
        "annotation_format": "YOLO",
        "license": "unknown",
        "annotation_coverage": {"male": None, "female": None, "FOCG": None},
        "quality_notes": "class set includes an unclear 'FOCG' label, worth inspecting before use",
        "status": "available",
    },
]


def seed_real_sources(run_artifacts_dir: Path) -> None:
    run_artifacts_dir = Path(run_artifacts_dir)
    run_artifacts_dir.mkdir(parents=True, exist_ok=True)
    (run_artifacts_dir / "sources.json").write_text(json.dumps(REAL_SOURCES, indent=2), encoding="utf-8")
    # No class_budget.json on purpose - dataset-agent's merge tool falls back
    # to the union of classes_covered (male, female, FOCG) when it's missing.
    print(f"Seeded REAL sources.json at {run_artifacts_dir} (no local files staged - nothing pre-downloaded).")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./run_artifacts_real_test")
    seed_real_sources(target)
