from deepagents import SubAgent


def build_sourcing_agent(roboflow_tools: list, kaggle_tools: list, web_search_tools: list) -> SubAgent:
    return {
        "name": "sourcing-agent",
        "description": (
            "Searches Roboflow Universe, Kaggle, and the web for datasets matching "
            "the class budget. Can be called multiple times - pass a specific "
            "class or coverage gap in the task prompt for a narrower, targeted "
            "follow-up search rather than a full re-run."
        ),
        "system_prompt": (
            "Search for public datasets and annotated data matching the classes "
            "you were asked about (all of them, or a specific subset named in your "
            "task). Prefer already-annotated sources. If sources.json already "
            "exists, read it first and append/update rather than overwrite. Write "
            "every candidate (source, license, image count, annotation coverage) "
            "to sources.json. Return only a short summary of what was found and "
            "what remains unlabeled or thin."
        ),
        "tools": [*roboflow_tools, *kaggle_tools, *web_search_tools],
    }
