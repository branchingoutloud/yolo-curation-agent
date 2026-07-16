from deepagents import SubAgent


def build_dataset_agent() -> SubAgent:
    return {
        "name": "dataset-agent",
        "description": (
            "Merges sourced/annotated data, dedupes near-identical images, splits "
            "train/val/test, and emits a YOLO-format data.yaml."
        ),
        "system_prompt": (
            "Read sources.json and annotation_manifest.json. Merge all available "
            "images and labels, remove near-duplicates, and split into "
            "train/val/test per the ratios in plan.md (consult the "
            "cv-dataset-curation skill for split/imbalance heuristics if unsure). "
            "Write dataset/data.yaml plus the underlying YOLO-format directory "
            "tree. Return only a short summary of final counts per split and "
            "class."
        ),
        "tools": [],
    }
