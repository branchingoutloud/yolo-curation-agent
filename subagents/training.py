import os

from deepagents import SubAgent

from tools.model_builder import build_model
from tools.training_runner import run_training

# TRAINING_AGENT_MODEL is an OPTIONAL override - unset by default, so
# training-agent inherits ORCHESTRATOR_MODEL. Its job is mostly mechanical -
# pick a model size per the yolo-model-selection skill, then call run_training
# once - so a lighter model is a reasonable override to set explicitly rather
# than spending the orchestrator's full-size budget here.
#
# run_training does the real work (real ultralytics training on the local GPU
# via the ultralytics Docker image - see tools/training_runner.py /
# tools/gpu_exec.py). This replaces the old, inert "backend": sandbox_backend
# approach: deepagents 0.6.12 ignores per-subagent backend overrides, so a
# plain-Python tool that shells out to Docker is how training actually runs
# (same pattern as dataset-agent's merge_and_split_dataset).


def build_training_agent() -> SubAgent:
    spec: SubAgent = {
        "name": "training-agent",
        "description": (
            "Selects a YOLO model size and runs real ultralytics training on the "
            "local GPU via run_training. Writes runs/train/metrics.json + best.pt."
        ),
        "system_prompt": (
            "Read dataset/data.yaml to confirm the dataset exists, and read the "
            "user-approved model size (from model_choice.json if present, else the "
            "size named in your task prompt). Consult the yolo-model-selection "
            "skill if the choice needs revisiting.\n\n"
            "Then call run_training exactly once with that model_size (one of "
            "yolo11n/yolo11s/yolo11m/yolo11l/yolo11x) and the chosen epochs/imgsz/"
            "batch. run_training trains on the GPU and writes runs/train/"
            "metrics.json and weights/best.pt itself - do not write those files by "
            "hand, and never paste raw ultralytics logs into your final message. "
            "If run_training reports a CUDA out-of-memory error, call it again with "
            "batch halved. Relay run_training's returned summary as your final "
            "message."
        ),
        "tools": [run_training],
    }
    model = build_model(os.environ.get("TRAINING_AGENT_MODEL"))
    if model is not None:
        spec["model"] = model
    return spec
