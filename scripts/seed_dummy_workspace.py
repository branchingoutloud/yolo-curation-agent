"""Seed a dummy /workspace/ (sources.json + staged source files) for testing
dataset-agent without real Roboflow/Kaggle credentials or network access.

Matches the sources.json contract sourcing-agent actually writes: one
"available" YOLO-format source (already staged, as if fetched) and one
"needs_annotation" source (deliberately left unstaged) so dataset-agent's
skip-and-report behavior gets exercised too, not just the happy path.

Usable standalone (`python scripts/seed_dummy_workspace.py [target_dir]`) or
imported (`seed_dummy_workspace(run_artifacts_dir)`) from another test script.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def seed_dummy_workspace(run_artifacts_dir: Path) -> None:
    run_artifacts_dir = Path(run_artifacts_dir)
    run_artifacts_dir.mkdir(parents=True, exist_ok=True)

    src0 = run_artifacts_dir / "sourced" / "0"
    (src0 / "images").mkdir(parents=True, exist_ok=True)
    (src0 / "labels").mkdir(parents=True, exist_ok=True)
    (src0 / "classes.txt").write_text("car\ntruck\n", encoding="utf-8")

    # Random-noise images (not flat colors - phash needs real structure to
    # distinguish "genuinely different" from "near-duplicate"; see the
    # dataset_builder test this mirrors).
    rng = np.random.default_rng(42)
    for i in range(4):
        img = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
        img.save(src0 / "images" / f"img_{i}.jpg")
        # alternate class per image so both car/truck get merged
        class_id = i % 2
        (src0 / "labels" / f"img_{i}.txt").write_text(f"{class_id} 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    # One exact duplicate of img_0 - dataset-agent's merge tool should drop it.
    Image.open(src0 / "images" / "img_0.jpg").save(src0 / "images" / "img_0_dup.jpg")
    (src0 / "labels" / "img_0_dup.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")

    sources = [
        {
            "source": "roboflow_universe",
            "dataset_id": "traffic-detection/traffic-detection-4",
            "url": "https://universe.roboflow.com/example/traffic-detection-4",
            "classes_covered": ["car", "truck"],
            "image_count": 5,
            "annotation_format": "YOLO",
            "license": "CC BY 4.0",
            "annotation_coverage": {"car": 3, "truck": 2},
            "quality_notes": "dummy synthetic data for dataset-agent testing",
            "status": "available",
        },
        {
            "source": "kaggle",
            "dataset_id": "andrewmvd/car-plate-detection",
            "url": "https://kaggle.com/datasets/andrewmvd/car-plate-detection",
            "classes_covered": ["bus"],
            "image_count": 300,
            "annotation_format": "Pascal VOC",
            "license": "CC0",
            "annotation_coverage": {"bus": 0},
            "quality_notes": "unannotated raw images - deliberately left unstaged",
            "status": "needs_annotation",
        },
    ]
    (run_artifacts_dir / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    (run_artifacts_dir / "class_budget.json").write_text(json.dumps(["car", "truck", "bus"]), encoding="utf-8")

    print(f"Seeded dummy workspace at {run_artifacts_dir}")
    print(f"  - sources.json: 1 available (roboflow, staged) + 1 needs_annotation (kaggle, unstaged)")
    print(f"  - sourced/0/: 5 images (1 exact duplicate), classes.txt")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./run_artifacts_test")
    seed_dummy_workspace(target)
