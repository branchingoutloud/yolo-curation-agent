from deepagents import SubAgent


def build_annotation_agent(roboflow_tools: list, zero_shot_tools: list) -> SubAgent:
    # No per-subagent sandbox override here: installed deepagents (0.6.12)
    # silently ignores a "backend" key on a SubAgent spec. execute() for this
    # subagent comes from the single backend passed to create_deep_agent —
    # see subagents/training.py's docstring for the full explanation.
    return {
        "name": "annotation-agent",
        "description": "Zero-shot pre-labels images for classes with no annotated data found.",
        "system_prompt": (
            "For each class flagged unlabeled in sources.json, run the zero-shot "
            "detector via execute() over the candidate images, push predictions "
            "into the Roboflow project as draft annotations for human review, "
            "and record results in annotation_manifest.json."
        ),
        "tools": [*roboflow_tools, *zero_shot_tools],
    }
