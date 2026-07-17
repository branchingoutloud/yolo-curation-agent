import os

from deepagents.middleware.subagents import CompiledSubAgent

from subagents.sandbox_subagent import build_sandbox_subagent
from tools.model_builder import build_model

# TRAINING_AGENT_MODEL is an OPTIONAL override - unset by default, so
# training-agent inherits ORCHESTRATOR_MODEL. Its job is mostly mechanical -
# pick a model size per the skill's rule, launch execute(), append status
# lines - so a lighter model is a reasonable override to set explicitly
# rather than spending the same budget as sourcing/dataset-agent's
# tool-calling load.

# Built via build_sandbox_subagent (subagents/sandbox_subagent.py), NOT a
# plain SubAgent dict with a "backend" field - that field is silently
# ignored by deepagents==0.6.12's subagent-building loop (see
# tools/dataset_builder.py's module docstring / CLAUDE.md's Known Stubs).
# This is instead its own separately-compiled create_deep_agent(backend=
# sandbox_backend, ...) graph wrapped as a CompiledSubAgent, which DOES get
# a real execute() tool. dataset/ and model_choice.json are copied into the
# sandbox before this subagent runs; runs/train/status.md and
# runs/train/metrics.json are copied back to real disk afterward (even on
# failure) so the orchestrator/UI can see partial progress.


def _resolve_model():
    # Unlike a declarative SubAgent dict (which deepagents' own subagent-
    # building loop falls back to ORCHESTRATOR_MODEL for automatically via
    # `spec.get("model", model)`), this subagent is its own independent
    # create_deep_agent(...) call - it has no automatic visibility into the
    # orchestrator's resolved model, so an unset TRAINING_AGENT_MODEL must
    # be resolved against ORCHESTRATOR_MODEL explicitly here, or it silently
    # falls back to deepagents' own built-in default (Anthropic) and fails
    # without ANTHROPIC_API_KEY set - the same gotcha documented in
    # CLAUDE.md for standalone test scripts.
    model_spec = os.environ.get("TRAINING_AGENT_MODEL") or os.environ.get(
        "ORCHESTRATOR_MODEL", "anthropic:claude-sonnet-5"
    )
    return build_model(model_spec)


def build_training_agent(sandbox_backend) -> CompiledSubAgent:
    return build_sandbox_subagent(
        name="training-agent",
        description="Selects a YOLO model size and runs ultralytics training in a GPU sandbox.",
        system_prompt=(
            "Read /workspace/dataset/data.yaml and the confirmed model size from "
            "/workspace/model_choice.json (consult the yolo-model-selection skill if the "
            "choice needs revisiting). Launch training via execute() - e.g. `yolo detect "
            "train data=/workspace/dataset/data.yaml model=<size>.pt ...`. Every few "
            "epochs, append a short status line to /workspace/runs/train/status.md - never "
            "print raw ultralytics logs to your final message. On completion, copy/rename "
            "ultralytics' actual output into /workspace/runs/train/metrics.json (and leave "
            "trained weights under /workspace/runs/train/weights/ - they stay in this "
            "sandbox for eval-agent to reuse directly, not downloaded to the shared "
            "workspace)."
        ),
        sandbox_backend=sandbox_backend,
        model=_resolve_model(),
        upload_paths=["/workspace/dataset", "/workspace/model_choice.json"],
        download_paths=["/workspace/runs/train/status.md", "/workspace/runs/train/metrics.json"],
    )
