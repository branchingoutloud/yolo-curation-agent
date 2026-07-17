import os

from deepagents import SubAgent

from tools.model_builder import build_model

# TRAINING_AGENT_MODEL is an OPTIONAL override - unset by default, so
# training-agent inherits ORCHESTRATOR_MODEL. Its job (once the sandbox-
# wiring bug in Known stubs is fixed) is mostly mechanical - pick a model
# size per the skill's rule, launch execute(), append status lines - so a
# lighter model is a reasonable override to set explicitly rather than
# spending the same budget as sourcing/dataset-agent's tool-calling load.


def build_training_agent(sandbox_backend) -> SubAgent:
    spec: SubAgent = {
        "name": "training-agent",
        "description": "Selects a YOLO model size and runs ultralytics training in a GPU sandbox.",
        "system_prompt": (
            "Read dataset/data.yaml and the confirmed model size from "
            "model_choice.json (consult the yolo-model-selection skill if the "
            "choice needs revisiting). Launch training via execute(). Every few "
            "epochs, append a short status line to runs/train/status.md - never "
            "print raw ultralytics logs to your final message. On completion "
            "write runs/train/metrics.json."
        ),
        "tools": [],
        "backend": sandbox_backend,  # Modal GPU sandbox
    }
    model = build_model(os.environ.get("TRAINING_AGENT_MODEL"))
    if model is not None:
        spec["model"] = model
    return spec
