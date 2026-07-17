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


# --- ALTERNATIVE: Modal sandbox approach (from feature/training_agent) ---
# Uses build_sandbox_subagent to create a CompiledSubAgent with a real
# execute() tool running in a Modal GPU sandbox instead of local Docker.
# To use this instead:
#   1. Uncomment the code below
#   2. Comment out the build_training_agent() above
#   3. Update agent.py to pass sandbox_backend to build_training_agent()
#
# from deepagents.middleware.subagents import CompiledSubAgent
# from subagents.sandbox_subagent import build_sandbox_subagent
#
# def _resolve_model():
#     # Unlike a declarative SubAgent dict (which deepagents' own subagent-
#     # building loop falls back to ORCHESTRATOR_MODEL for automatically via
#     # `spec.get("model", model)`), this subagent is its own independent
#     # create_deep_agent(...) call - it has no automatic visibility into the
#     # orchestrator's resolved model, so an unset TRAINING_AGENT_MODEL must
#     # be resolved against ORCHESTRATOR_MODEL explicitly here, or it silently
#     # falls back to deepagents' own built-in default (Anthropic) and fails
#     # without ANTHROPIC_API_KEY set.
#     model_spec = os.environ.get("TRAINING_AGENT_MODEL") or os.environ.get(
#         "ORCHESTRATOR_MODEL", "anthropic:claude-sonnet-5"
#     )
#     return build_model(model_spec)
#
#
# def build_training_agent(sandbox_backend) -> CompiledSubAgent:
#     return build_sandbox_subagent(
#         name="training-agent",
#         description="Selects a YOLO model size and runs ultralytics training in a GPU sandbox.",
#         system_prompt=(
#             "Read /workspace/dataset/data.yaml and the confirmed model size from "
#             "/workspace/model_choice.json (consult the yolo-model-selection skill if the "
#             "choice needs revisiting). Launch training via execute() - e.g. `yolo detect "
#             "train data=/workspace/dataset/data.yaml model=<size>.pt ...`. Every few "
#             "epochs, append a short status line to /workspace/runs/train/status.md - never "
#             "print raw ultralytics logs to your final message. On completion, copy/rename "
#             "ultralytics' actual output into /workspace/runs/train/metrics.json (and leave "
#             "trained weights under /workspace/runs/train/weights/ - they stay in this "
#             "sandbox for eval-agent to reuse directly, not downloaded to the shared "
#             "workspace)."
#         ),
#         sandbox_backend=sandbox_backend,
#         model=_resolve_model(),
#         upload_paths=["/workspace/dataset", "/workspace/model_choice.json"],
#         download_paths=["/workspace/runs/train/status.md", "/workspace/runs/train/metrics.json"],
#     )
