import os

from deepagents import SubAgent

from tools.model_builder import build_model

# EVAL_AGENT_MODEL is an OPTIONAL override - unset by default, so eval-agent
# inherits ORCHESTRATOR_MODEL. Diagnosing *why* a class underperforms (not
# just which metric is low) is the one genuinely reasoning-heavy step here,
# but the mechanical parts (confusion matrix via execute(), writing the two
# output files) don't need the orchestrator's full-size model - a lighter
# override is a reasonable default to set explicitly, same as training-agent.


def build_eval_agent(sandbox_backend) -> SubAgent:
    spec: SubAgent = {
        "name": "eval-agent",
        "description": (
            "Analyzes a completed training run: confusion matrix, per-class "
            "metrics, and a best-guess diagnosis for each weak class (too few "
            "images, high visual variability, likely label noise, or a class the "
            "zero-shot annotator struggled with). Does NOT decide the next "
            "action - that's the orchestrator's call based on this diagnosis."
        ),
        "system_prompt": (
            "Read runs/train/metrics.json and the trained weights. Use the "
            "`supervision` library via execute() to build a confusion matrix and "
            "sample failure crops. Consult the cv-eval-and-iteration skill for "
            "how to read mAP50/mAP50-95 and confusion-matrix patterns. For each "
            "underperforming class, inspect its sample count, source annotation "
            "quality, and failure crops, and record a specific likely cause - not "
            "just the metric. Write eval_report.md (human-readable) and "
            "weak_classes.json (structured: class name, metric, sample count, "
            "likely cause) so the orchestrator can decide whether to re-source, "
            "re-annotate, or just retune."
        ),
        "tools": [],
        "backend": sandbox_backend,  # reuse - weights are already there
    }
    model = build_model(os.environ.get("EVAL_AGENT_MODEL"))
    if model is not None:
        spec["model"] = model
    return spec
