import os

from deepagents.middleware.subagents import CompiledSubAgent

from subagents.sandbox_subagent import build_sandbox_subagent
from tools.model_builder import build_model

# EVAL_AGENT_MODEL is an OPTIONAL override - unset by default, so eval-agent
# inherits ORCHESTRATOR_MODEL. Diagnosing *why* a class underperforms (not
# just which metric is low) is the one genuinely reasoning-heavy step here,
# but the mechanical parts (confusion matrix via execute(), writing the two
# output files) don't need the orchestrator's full-size model - a lighter
# override is a reasonable default to set explicitly, same as training-agent.

# Built via build_sandbox_subagent, same reasoning as training-agent (see
# subagents/training.py) - a separately-compiled create_deep_agent(backend=
# sandbox_backend) graph wrapped as a CompiledSubAgent, not a plain SubAgent
# dict (whose "backend" field deepagents==0.6.12 silently ignores). No
# upload_paths here: `sandbox_backend` is the SAME ModalSandbox instance
# passed to build_training_agent (see agent.py/backends/sandboxes.py's
# process-lifetime singleton), so the dataset and trained weights training-
# agent already staged there are still present - re-uploading them would be
# redundant. eval_report.md/weak_classes.json are copied back to real disk
# so the orchestrator/UI can read them.


def _resolve_model():
    # See subagents/training.py's _resolve_model - same reasoning: this is
    # its own independent create_deep_agent(...) call, not a declarative
    # SubAgent dict, so it doesn't automatically inherit ORCHESTRATOR_MODEL
    # the way deepagents' own subagent-building loop would.
    model_spec = os.environ.get("EVAL_AGENT_MODEL") or os.environ.get(
        "ORCHESTRATOR_MODEL", "anthropic:claude-sonnet-5"
    )
    return build_model(model_spec)


def build_eval_agent(sandbox_backend) -> CompiledSubAgent:
    return build_sandbox_subagent(
        name="eval-agent",
        description=(
            "Analyzes a completed training run: confusion matrix, per-class "
            "metrics, and a best-guess diagnosis for each weak class (too few "
            "images, high visual variability, likely label noise, or a class the "
            "zero-shot annotator struggled with). Does NOT decide the next "
            "action - that's the orchestrator's call based on this diagnosis."
        ),
        system_prompt=(
            "Read /workspace/runs/train/metrics.json and the trained weights under "
            "/workspace/runs/train/weights/ (already in this sandbox - training-agent "
            "just ran here). Use the `supervision` library via execute() to build a "
            "confusion matrix and sample failure crops. Consult the cv-eval-and-iteration "
            "skill for how to read mAP50/mAP50-95 and confusion-matrix patterns. For each "
            "underperforming class, inspect its sample count, source annotation quality, "
            "and failure crops, and record a specific likely cause - not just the metric. "
            "Write /workspace/eval_report.md (human-readable) and /workspace/"
            "weak_classes.json (structured: class name, metric, sample count, likely "
            "cause) so the orchestrator can decide whether to re-source, re-annotate, or "
            "just retune."
        ),
        sandbox_backend=sandbox_backend,
        model=_resolve_model(),
        upload_paths=[],
        download_paths=["/workspace/eval_report.md", "/workspace/weak_classes.json"],
    )
