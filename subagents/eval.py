from deepagents import SubAgent


def build_eval_agent(sandbox_backend) -> SubAgent:
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
        # reuse training sandbox (weights already there); omitted until configured
        **({"backend": sandbox_backend} if sandbox_backend is not None else {}),
    }
