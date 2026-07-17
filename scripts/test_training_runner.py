"""Smoke-test the GPU training/eval runner tools WITHOUT the LLM.

This exercises tools/training_runner.py + tools/eval_runner.py directly (the
same "call .invoke() against seeded files" pattern used for
merge_and_split_dataset), so it needs no model call. It DOES need the real
runtime, though: Docker + the NVIDIA Container Toolkit + a GPU + the ultralytics
image, i.e. run this ON the GPU server, not on the dev laptop.

Prereq: a dataset must already exist at RUN_ARTIFACTS_DIR/dataset/data.yaml.
Build a throwaway one with NO model call (seed + a direct merge invoke):
    uv run python scripts/seed_dummy_workspace.py ./run_artifacts_test
    RUN_ARTIFACTS_DIR=./run_artifacts_test uv run python -c \
      "from tools.dataset_builder import merge_and_split_dataset; print(merge_and_split_dataset.invoke({}))"

Then run this against the same workspace:
    RUN_ARTIFACTS_DIR=./run_artifacts_test uv run python scripts/test_training_runner.py

(Or point RUN_ARTIFACTS_DIR at any real run that already has dataset/.)
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends.project_backend import RUN_ARTIFACTS_DIR  # noqa: E402
from tools.eval_runner import run_eval  # noqa: E402
from tools.training_runner import run_training  # noqa: E402


def main() -> int:
    data_yaml = Path(RUN_ARTIFACTS_DIR).resolve() / "dataset" / "data.yaml"
    print(f"RUN_ARTIFACTS_DIR = {RUN_ARTIFACTS_DIR}")
    print(f"dataset data.yaml = {data_yaml}  exists={data_yaml.exists()}")
    if not data_yaml.exists():
        print(
            "\nNo dataset found. Build one first (see this file's docstring), "
            "then re-run."
        )
        return 1

    print("\n=== run_training (tiny: yolo11n, 1 epoch, imgsz 320, batch 4) ===")
    train_result = run_training.invoke(
        {"model_size": "yolo11n", "epochs": 1, "imgsz": 320, "batch": 4}
    )
    print(train_result)

    best = Path(RUN_ARTIFACTS_DIR).resolve() / "runs" / "train" / "weights" / "best.pt"
    metrics = Path(RUN_ARTIFACTS_DIR).resolve() / "runs" / "train" / "metrics.json"
    print(f"\nbest.pt exists    = {best.exists()}")
    print(f"metrics.json exists = {metrics.exists()}")
    if not (best.exists() and metrics.exists()):
        print("Training did not produce expected artifacts - see output above.")
        return 1

    print("\n=== run_eval (val split) ===")
    eval_result = run_eval.invoke({"imgsz": 320, "split": "val"})
    print(eval_result)

    report = Path(RUN_ARTIFACTS_DIR).resolve() / "eval_report.md"
    weak = Path(RUN_ARTIFACTS_DIR).resolve() / "weak_classes.json"
    print(f"\neval_report.md exists    = {report.exists()}")
    print(f"weak_classes.json exists = {weak.exists()}")
    return 0 if (report.exists() and weak.exists()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
