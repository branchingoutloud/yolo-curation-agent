import os

from deepagents import SubAgent

from tools.model_builder import build_model
from tools.eval_runner import run_eval

# EVAL_AGENT_MODEL is an OPTIONAL override - unset by default, so eval-agent
# inherits ORCHESTRATOR_MODEL. Diagnosing *why* a class underperforms (not just
# which metric is low) is the one genuinely reasoning-heavy step here; the
# mechanical parts (running validation, writing the two output files) are done
# by run_eval, so a lighter override is reasonable, same as training-agent.
#
# run_eval does the real work (ultralytics validation on the local GPU via the
# ultralytics Docker image - see tools/eval_runner.py / tools/gpu_exec.py),
# replacing the old inert "backend": sandbox_backend approach.


def build_eval_agent() -> SubAgent:
    spec: SubAgent = {
        "name": "eval-agent",
        "description": (
            "Evaluates the trained model on the local GPU (per-class metrics + "
            "confusion matrix) and diagnoses each weak class. Does NOT decide the "
            "next action - that's the orchestrator's call based on this diagnosis."
        ),
        "system_prompt": (
            "Call run_eval exactly once. It reads runs/train/weights/best.pt and "
            "dataset/data.yaml, runs validation on the GPU, and writes "
            "eval_report.md and weak_classes.json itself - do not write those "
            "files by hand.\n\n"
            "Consult the cv-eval-and-iteration skill for how to read mAP50/"
            "mAP50-95 and confusion-matrix patterns. Relay run_eval's summary as "
            "your final message, adding a one-line read per weak class on the "
            "likely cause (too few images, high visual variability, label noise, "
            "or confusion with a specific other class). Do NOT decide whether to "
            "re-source, re-annotate, or retrain - that is the orchestrator's call."
        ),
        "tools": [run_eval],
    }
    model = build_model(os.environ.get("EVAL_AGENT_MODEL"))
    if model is not None:
        spec["model"] = model
    return spec
