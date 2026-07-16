from deepagents import SubAgent


def build_training_agent(sandbox_backend) -> SubAgent:
    return {
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
