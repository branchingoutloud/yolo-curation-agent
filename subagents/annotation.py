from deepagents import SubAgent


def build_annotation_agent(roboflow_tools: list, zero_shot_tools: list, sandbox_backend) -> SubAgent:
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
        # per-subagent sandbox override; omitted until one is configured
        **({"backend": sandbox_backend} if sandbox_backend is not None else {}),
    }
