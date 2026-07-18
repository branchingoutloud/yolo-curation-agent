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
        "dataset_id": "data-o6svn/car-92bpq",
        "url": "https://universe.roboflow.com/data-o6svn/car-92bpq",
        "classes_covered": ["car"],
        "image_count": 154,
        "annotation_format": "YOLO",
        "license": "CC BY 4.0",
        "annotation_coverage": {"car": None},
        "quality_notes": "largest of 5 seeded car sources - should be selected under the max_sources=3 cap",
        "status": "available",
    },
    {
        "source": "roboflow_universe",
        "dataset_id": "radhakrishnan-v-6ax3b/car-detection-hwbmw",
        "url": "https://universe.roboflow.com/radhakrishnan-v-6ax3b/car-detection-hwbmw",
        "classes_covered": ["car", "cars"],
        "image_count": 106,
        "annotation_format": "YOLO",
        "license": "CC BY 4.0",
        "annotation_coverage": {"car": None, "cars": None},
        "quality_notes": "2nd largest of 5 seeded car sources - should be selected under the max_sources=3 cap",
        "status": "available",
    },
    {
        "source": "roboflow_universe",
        "dataset_id": "imc/car-dwybi",
        "url": "https://universe.roboflow.com/imc/car-dwybi",
        "classes_covered": ["Car", "Cars"],
        "image_count": 105,
        "annotation_format": "YOLO",
        "license": "CC BY 4.0",
        "annotation_coverage": {"Car": None, "Cars": None},
        "quality_notes": "3rd largest of 5 seeded car sources - should be selected under the max_sources=3 cap",
        "status": "available",
    },
    {
        "source": "roboflow_universe",
        "dataset_id": "main-k1c35/car-detection-afv4z",
        "url": "https://universe.roboflow.com/main-k1c35/car-detection-afv4z",
        "classes_covered": ["car", "0", "4"],
        "image_count": 100,
        "annotation_format": "YOLO",
        "license": "CC BY 4.0",
        "annotation_coverage": {"car": None},
        "quality_notes": "4th largest of 5 seeded car sources - should be EXCLUDED by the max_sources=3 cap",
        "status": "available",
    },
    {
        "source": "roboflow_universe",
        "dataset_id": "train-zwdjq/car-detection-fj7cr",
        "url": "https://universe.roboflow.com/train-zwdjq/car-detection-fj7cr",
        "classes_covered": ["car", "objects"],
        "image_count": 34,
        "annotation_format": "YOLO",
        "license": "CC BY 4.0",
        "annotation_coverage": {"car": None},
        "quality_notes": "smallest of 5 seeded car sources - should be EXCLUDED by the max_sources=3 cap",
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
