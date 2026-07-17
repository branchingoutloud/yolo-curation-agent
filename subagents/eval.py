from deepagents import SubAgent


def build_eval_agent() -> SubAgent:
    # No per-subagent sandbox override here: installed deepagents (0.6.12)
    # silently ignores a "backend" key on a SubAgent spec. execute() for this
    # subagent comes from the single backend passed to create_deep_agent —
    # see subagents/training.py's docstring for the full explanation.
    return {
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
    }
