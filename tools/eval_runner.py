"""`run_eval` tool for eval-agent - real per-class eval on the local GPU.

Same shape and rationale as tools/training_runner.py: plain Python the subagent
calls once, shelling out to the ultralytics GPU container (tools/gpu_exec.py).
The eval_driver reads the trained weights + dataset, runs validation, and writes
eval_report.md + weak_classes.json itself. This tool reads weak_classes.json
back and returns a short summary; it deliberately does NOT decide the next
action - that stays the orchestrator's call.
"""

from __future__ import annotations

import json
import os

from langchain_core.tools import tool

from tools.gpu_exec import GpuExecError, run_driver
from tools.workspace_paths import workspace_path as _workspace_path

_DEFAULT_TIMEOUT = int(os.environ.get("YOLO_EVAL_TIMEOUT", str(30 * 60)))
_DEVICE = os.environ.get("YOLO_TRAIN_DEVICE", "0")
_VALID_SPLITS = {"train", "val", "test"}


@tool
def run_eval(imgsz: int = 640, split: str = "val") -> str:
    """Evaluate the trained model on the local GPU and diagnose weak classes.

    Call this once after training. It reads /workspace/runs/train/weights/best.pt
    and /workspace/dataset/data.yaml, runs validation on `split` (one of
    train/val/test), and writes /workspace/eval_report.md and
    /workspace/weak_classes.json itself. Returns a short summary: overall
    mAP50/mAP50-95 and the weak classes with their likely cause. Do NOT write
    those files by hand, and do NOT decide whether to re-source/re-train - report
    the diagnosis and let the orchestrator decide.
    """
    chosen_split = split.strip().lower()
    if chosen_split not in _VALID_SPLITS:
        return f"Error: split must be one of {sorted(_VALID_SPLITS)} (got {split!r})."
    if imgsz < 32:
        return f"Error: imgsz too small ({imgsz})."

    try:
        best = _workspace_path("/workspace/runs/train/weights/best.pt")
        data_yaml = _workspace_path("/workspace/dataset/data.yaml")
    except ValueError as exc:
        return f"Error: {exc}"
    if not best.exists():
        return (
            "Error: /workspace/runs/train/weights/best.pt not found - training-agent "
            "must train a model before eval."
        )
    if not data_yaml.exists():
        return "Error: /workspace/dataset/data.yaml not found - dataset-agent must run first."

    try:
        output = run_driver(
            "eval_driver.py",
            [str(imgsz), _DEVICE, chosen_split],
            timeout=_DEFAULT_TIMEOUT,
        )
    except GpuExecError as exc:
        return f"Eval failed: {exc}"

    try:
        weak_path = _workspace_path("/workspace/weak_classes.json")
    except ValueError as exc:
        return f"Error: {exc}"
    if not weak_path.exists():
        return f"Eval container finished but wrote no weak_classes.json. Output tail:\n{output[-2000:]}"

    try:
        data = json.loads(weak_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return f"Eval finished but weak_classes.json is unreadable: {exc}"

    weak = data.get("weak_classes", [])
    if weak:
        weak_str = "; ".join(f"{w.get('class')} (mAP50={w.get('map50')}, {w.get('likely_cause')})" for w in weak)
    else:
        weak_str = "none - all classes above threshold"
    return (
        f"Eval complete on '{chosen_split}' split. "
        f"Overall mAP50={data.get('overall_map50')}, mAP50-95={data.get('overall_map50_95')}. "
        f"Weak classes: {weak_str}. "
        "Wrote /workspace/eval_report.md and /workspace/weak_classes.json."
    )
