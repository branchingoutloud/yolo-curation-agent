"""`run_training` tool for training-agent - real YOLO training on the local GPU.

Like tools/dataset_builder.py, this is plain Python the subagent calls once, not
a prompted execute(). It shells out to the ultralytics GPU container (see
tools/gpu_exec.py), which does the actual torch training against the local GPU,
then reads back the metrics.json the driver wrote and returns a short summary.

Why this shape instead of giving training-agent an `execute` tool: the installed
deepagents (0.6.12) binds every subagent's filesystem/execute tools to the single
top-level backend passed to create_deep_agent, so a per-subagent sandbox backend
is inert (see tools/dataset_builder.py's docstring). Running training as a
plain-Python tool that shells out to Docker sidesteps that entirely - the same
pattern that already works for merge_and_split_dataset - and keeps torch out of
the orchestrator venv.

All artifacts (weights/best.pt, results.csv, metrics.json) land under
RUN_ARTIFACTS_DIR/runs/train on the host, inspectable after the run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_core.tools import tool

from tools.gpu_exec import GpuExecError, run_driver
from tools.workspace_paths import workspace_path as _workspace_path

_VALID_SIZES = {"yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"}
_DEFAULT_TIMEOUT = int(os.environ.get("YOLO_TRAIN_TIMEOUT", str(3 * 3600)))
_DEVICE = os.environ.get("YOLO_TRAIN_DEVICE", "0")


def _fmt(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


@tool
def run_training(
    model_size: str = "yolo11m",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
) -> str:
    """Train a YOLO detection model on the merged dataset using the local GPU.

    Call this once, after the user has approved a model size. `model_size` must
    be one of yolo11n/yolo11s/yolo11m/yolo11l/yolo11x (consult the
    yolo-model-selection skill to choose). It trains on
    /workspace/dataset/data.yaml (built by dataset-agent), writes
    /workspace/runs/train/metrics.json and weights/best.pt itself, and returns a
    short summary of final mAP50 / mAP50-95 / per-class AP. Do NOT write those
    files by hand and do NOT paste raw training logs into your reply. If the
    result mentions a CUDA out-of-memory error, call this again with `batch`
    halved.
    """
    size = model_size.strip().lower()
    if size not in _VALID_SIZES:
        return f"Error: model_size must be one of {sorted(_VALID_SIZES)} (got {model_size!r})."
    if epochs < 1 or imgsz < 32 or batch < 1:
        return f"Error: invalid args (epochs={epochs}, imgsz={imgsz}, batch={batch})."

    try:
        data_yaml = _workspace_path("/workspace/dataset/data.yaml")
        train_dir = _workspace_path("/workspace/runs/train")
    except ValueError as exc:
        return f"Error: {exc}"
    if not data_yaml.exists():
        return (
            "Error: /workspace/dataset/data.yaml not found - dataset-agent must "
            "build the dataset before training."
        )

    train_dir.mkdir(parents=True, exist_ok=True)
    status_path = train_dir / "status.md"
    status_path.write_text(
        f"# Training status\n\n- Started: {size}, epochs={epochs}, imgsz={imgsz}, "
        f"batch={batch}, device={_DEVICE}\n",
        encoding="utf-8",
    )

    try:
        output = run_driver(
            "train_driver.py",
            [size, str(epochs), str(imgsz), str(batch), _DEVICE],
            timeout=_DEFAULT_TIMEOUT,
        )
    except GpuExecError as exc:
        status_path.write_text(f"# Training status\n\n- FAILED: {exc}\n", encoding="utf-8")
        return f"Training failed: {exc}"

    metrics_path = train_dir / "metrics.json"
    if not metrics_path.exists():
        return (
            "Training container finished but wrote no metrics.json (likely an "
            f"error late in the run). Output tail:\n{output[-2000:]}"
        )

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return f"Training finished but metrics.json is unreadable: {exc}"

    status_path.write_text(
        f"# Training status\n\n- Done: {size}. mAP50={_fmt(metrics.get('map50'))}, "
        f"mAP50-95={_fmt(metrics.get('map50_95'))}\n",
        encoding="utf-8",
    )

    per_class = metrics.get("per_class", {})
    per_class_str = ", ".join(f"{name}: mAP50={_fmt(v.get('map50'))}" for name, v in per_class.items()) or "n/a"
    return (
        f"Training complete ({size}, {epochs} epochs, imgsz {imgsz}, batch {batch}). "
        f"mAP50={_fmt(metrics.get('map50'))}, mAP50-95={_fmt(metrics.get('map50_95'))}, "
        f"precision={_fmt(metrics.get('precision'))}, recall={_fmt(metrics.get('recall'))}. "
        f"Per-class mAP50: {per_class_str}. "
        "Weights at /workspace/runs/train/weights/best.pt; metrics written to "
        "/workspace/runs/train/metrics.json."
    )
