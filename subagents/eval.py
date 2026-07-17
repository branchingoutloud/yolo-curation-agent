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


# --- ALTERNATIVE: Modal sandbox approach (from feature/training_agent) ---
# Uses build_sandbox_subagent to create a CompiledSubAgent that reuses the
# training sandbox (same ModalSandbox instance, so trained weights are
# already local — no re-upload needed). To use this instead:
#   1. Uncomment the code below
#   2. Comment out the build_eval_agent() above
#   3. Update agent.py to pass sandbox_backend to build_eval_agent()
#
# from deepagents.middleware.subagents import CompiledSubAgent
# from subagents.sandbox_subagent import build_sandbox_subagent
#
# def _resolve_model():
#     # See subagents/training.py's _resolve_model - same reasoning: this is
#     # its own independent create_deep_agent(...) call, not a declarative
#     # SubAgent dict, so it doesn't automatically inherit ORCHESTRATOR_MODEL
#     # the way deepagents' own subagent-building loop would.
#     model_spec = os.environ.get("EVAL_AGENT_MODEL") or os.environ.get(
#         "ORCHESTRATOR_MODEL", "anthropic:claude-sonnet-5"
#     )
#     return build_model(model_spec)
#
#
# def build_eval_agent(sandbox_backend) -> CompiledSubAgent:
#     return build_sandbox_subagent(
#         name="eval-agent",
#         description=(
#             "Analyzes a completed training run: confusion matrix, per-class "
#             "metrics, and a best-guess diagnosis for each weak class (too few "
#             "images, high visual variability, likely label noise, or a class the "
#             "zero-shot annotator struggled with). Does NOT decide the next "
#             "action - that's the orchestrator's call based on this diagnosis."
#         ),
#         system_prompt=(
#             "Read /workspace/runs/train/metrics.json and the trained weights under "
#             "/workspace/runs/train/weights/ (already in this sandbox - training-agent "
#             "just ran here). Use the `supervision` library via execute() to build a "
#             "confusion matrix and sample failure crops. Consult the cv-eval-and-iteration "
#             "skill for how to read mAP50/mAP50-95 and confusion-matrix patterns. For each "
#             "underperforming class, inspect its sample count, source annotation quality, "
#             "and failure crops, and record a specific likely cause - not just the metric. "
#             "Write /workspace/eval_report.md (human-readable) and /workspace/"
#             "weak_classes.json (structured: class name, metric, sample count, likely "
#             "cause) so the orchestrator can decide whether to re-source, re-annotate, or "
#             "just retune."
#         ),
#         sandbox_backend=sandbox_backend,
#         model=_resolve_model(),
#         upload_paths=[],
#         download_paths=["/workspace/eval_report.md", "/workspace/weak_classes.json"],
#     )
