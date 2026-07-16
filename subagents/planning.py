from deepagents import SubAgent


def build_planning_agent() -> SubAgent:
    return {
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
            "concrete heuristics. Write your reasoning and final numbers to "
            "plan.md and class_budget.json. Return only a one-paragraph summary."
        ),
        "tools": [],  # filesystem tools are auto-attached by the harness
    }
