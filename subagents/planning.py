import os

from deepagents import SubAgent

from tools.model_builder import build_model
from tools.planning_builder import write_plan

# PLANNING_AGENT_MODEL is an OPTIONAL override - unset by default, so
# planning-agent inherits ORCHESTRATOR_MODEL like most subagents. It only
# ever calls write_plan once, but the reasoning behind per-class variability
# tiers/budgets benefits from the stronger model the orchestrator already
# runs, so inheriting (rather than trimming down to something smaller) is
# the right default here.


def build_planning_agent() -> SubAgent:
    spec: SubAgent = {
        "name": "planning-agent",
        "description": (
            "Given target classes, use case, and deployment target, estimates "
            "images-per-class needed and proposes a dataset size/split. Use "
            "this before any data is sourced, and again later if sourced "
            "coverage doesn't match the original estimate."
        ),
        "system_prompt": (
            "You are a dataset-planning specialist. Lower intra-class visual "
            "variability (e.g. a single fixed logo) needs far fewer images than "
            "high-variability classes (e.g. 'pedestrian' across poses/lighting). "
            "Consult the yolo-model-selection and cv-dataset-curation skills for "
            "concrete heuristics before deciding numbers.\n\n"
            "Call write_plan exactly once with your full reasoning: for each "
            "class, its name, a variability_tier (one of very_low/moderate/high/"
            "very_high, per the cv-dataset-curation skill's table), your proposed "
            "images_per_class, and a one-line rationale. write_plan writes both "
            "plan.md and class_budget.json itself and checks your numbers "
            "against the skill's floors and imbalance thresholds, returning any "
            "warnings - address those before finalizing rather than ignoring "
            "them. Relay its returned summary as your final message; you do not "
            "need to write plan.md or class_budget.json by hand."
        ),
        "tools": [write_plan],
    }
    model = build_model(os.environ.get("PLANNING_AGENT_MODEL"))
    if model is not None:
        spec["model"] = model
    return spec
