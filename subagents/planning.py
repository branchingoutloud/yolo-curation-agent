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

# TRAINING_BACKEND=local (backends/sandboxes.py's default) means there is no
# GPU/cloud sourcing infra behind this run at all - sourcing-agent has to
# actually find and catalog every image via free Roboflow/Kaggle/web search,
# and dataset-agent/training-agent both run on this one CPU machine. The
# cv-dataset-curation skill's floors (800-1500 images for high/very_high
# variability classes) are real production guidance, not something this setup
# can source and train on in one sitting - a live "car" run asked for on this
# basis produced an unworkable 2000-image plan and, downstream, a sourcing
# run that fabricated fake dataset entries (KITTI/Cityscapes/Waymo/
# OpenImages - see subagents/sourcing.py) rather than admit it couldn't find
# that many real, searchable sources. Ask for a small demo-scale budget by
# default instead; only use the skill's full production floors if the user
# explicitly asks for a production-quality model rather than a working v1.
_LOCAL_DEMO_NOTE = (
    "\n\nIMPORTANT - this pipeline is currently running end-to-end on a single "
    "local CPU machine with no GPU/cloud budget and no bulk-download "
    "infrastructure (TRAINING_BACKEND=local). Unless the user has explicitly "
    "asked for a production-quality model, treat this as a smoke test proving "
    "the pipeline works, not a real accuracy target - propose a SMALL demo-"
    "scale budget (roughly 20-40 images per class total) rather than the "
    "cv-dataset-curation skill's full production floors (800+ for a "
    "high-variability class). write_plan will warn that this is below the "
    "skill's floor for that tier - that warning is expected here, note in your "
    "rationale that it's an intentional smoke-test reduction, not an error. A "
    "small, honestly-sourced dataset that actually completes the full "
    "sourcing -> training -> eval flow is more useful right now than a large "
    "plan that can't realistically be fulfilled by real search results."
)


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
            "concrete heuristics before deciding numbers."
            + (_LOCAL_DEMO_NOTE if os.environ.get("TRAINING_BACKEND", "local") == "local" else "")
            + "\n\nCall write_plan exactly once with your full reasoning: for each "
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
