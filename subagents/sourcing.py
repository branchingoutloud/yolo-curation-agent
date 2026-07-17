import os

from deepagents import SubAgent

from tools.model_builder import build_model
from tools.sourcing_builder import append_sources

# SOURCING_AGENT_MODEL is an OPTIONAL override - unset by default, so
# sourcing-agent inherits ORCHESTRATOR_MODEL. This one is tool-call-heavy
# (Roboflow/Kaggle/Tavily search, then append_sources) and previously hit
# `groq.APIError: Failed to call a function` on a smaller model - worth
# keeping on the orchestrator's (larger) model rather than trimming down.

# sourcing-agent only *catalogs* candidates (source, license, image count,
# classes covered) - dataset-agent is the one that actually fetches/forks/
# exports (see subagents/dataset.py). Roboflow's MCP server exposes 100+
# tools; binding all of them bloats the prompt for no benefit here and risks
# the same free-tier token-limit issues dataset-agent hit with an untrimmed
# list (see CLAUDE.md's Architecture/Testing sections) - this keeps
# sourcing-agent to search/discovery tools only.
_ROBOFLOW_TOOL_ALLOWLIST = {
    "universe_search",
}

# Kaggle's real MCP server (https://www.kaggle.com/mcp) exposes ~70 tools
# spanning its whole product surface (competitions, notebooks, forums,
# hackathons) - same bloat problem as Roboflow's 107, trimmed to just dataset
# discovery/metadata (not download_dataset/upload_dataset_file - fetching is
# dataset-agent's job, see subagents/dataset.py).
#
# Both allowlists here are cut to the bare minimum (1 tool each) because even
# the "reasonable" 4+4 first pass landed at 9781-12939 tokens against Groq's
# 8000-12000 TPM free-tier caps (gpt-oss-120b / llama-3.3-70b) - deepagents'
# own middleware (filesystem tools, todo tool, boilerplate instructions) adds
# a large, roughly fixed per-request floor on top of whatever custom tools are
# bound, so trimming further only buys a small margin. If even this doesn't
# clear Groq's free tier, that's a provider-limit problem, not a tool-count
# one - see CLAUDE.md's Testing section and wait for Ollama (no such limit)
# rather than continuing to trim.
_KAGGLE_TOOL_ALLOWLIST = {
    "search_datasets",
}


def _filter_roboflow_tools(roboflow_tools: list) -> list:
    return [t for t in roboflow_tools if getattr(t, "name", None) in _ROBOFLOW_TOOL_ALLOWLIST]


def _filter_kaggle_tools(kaggle_tools: list) -> list:
    return [t for t in kaggle_tools if getattr(t, "name", None) in _KAGGLE_TOOL_ALLOWLIST]


def build_sourcing_agent(roboflow_tools: list, kaggle_tools: list, web_search_tools: list) -> SubAgent:
    spec: SubAgent = {
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
            "task). Prefer already-annotated sources.\n\n"
            "For every candidate you find, call append_sources once with the full "
            "list of new/updated entries - it reads sources.json first and updates-"
            "or-appends by (source, dataset_id), so you never need to read the file "
            "yourself or worry about overwriting what a previous round already "
            "found. Don't pass sources_json_path unless you have a real reason to "
            "point at a different file - its default is already correct. Each entry "
            "needs: source, dataset_id, url, classes_covered, "
            "image_count (a single total int for the whole dataset - do NOT put a "
            "per-class breakdown here, that's what annotation_coverage is for), "
            "annotation_format, license, annotation_coverage (a {class_name: count} "
            "dict), quality_notes, status (\"available\" if it has real annotations "
            "already, \"needs_annotation\" if it's raw/unlabeled images only - "
            "don't mark something \"available\" just because it exists).\n\n"
            "Return only a short summary of what was found and what remains "
            "unlabeled or thin - relay append_sources' returned summary, adding "
            "your own read on which classes still need work "
            "(e.g. \"'bus' unlabeled - 300 raw images found, 0 annotated; "
            "recommend routing to annotation once that's available\"). This summary "
            "is the routing signal the orchestrator and dataset-agent act on directly, "
            "not something they re-verify by re-reading sources.json themselves - so "
            "never mention a dataset/source you did not actually pass to append_sources "
            "in this same turn, even as a `next step` suggestion. If you don't have a "
            "search tool for some provider (e.g. no HuggingFace tool is bound here), "
            "say you didn't search it rather than describing results from it."
        ),
        "tools": [
            *_filter_roboflow_tools(roboflow_tools),
            *_filter_kaggle_tools(kaggle_tools),
            *web_search_tools,
            append_sources,
        ],
    }
    model = build_model(os.environ.get("SOURCING_AGENT_MODEL"))
    if model is not None:
        spec["model"] = model
    return spec
