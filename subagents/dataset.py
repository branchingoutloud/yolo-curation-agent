"""dataset-agent: fetches sourced data and merges/dedupes/splits it into a
YOLO-format dataset. Runs directly after sourcing-agent - annotation-agent is
temporarily excluded from the active roster (see agent.py), so any
sources.json entry with `status != "available"` is left pending rather than
routed anywhere; dataset-agent just reports it back to the orchestrator.
"""

import os

from deepagents import SubAgent

from tools.dataset_builder import download_and_extract, list_qualifying_sources, merge_and_split_dataset
from tools.model_builder import build_model

# DATASET_AGENT_MODEL is an OPTIONAL override on top of ORCHESTRATOR_MODEL -
# every other subagent (planning/sourcing/training/eval) already inherits
# ORCHESTRATOR_MODEL automatically for free (deepagents' own subagent-building
# loop does `spec.get("model", model)` - see graph.py), so dataset-agent
# falling back to ORCHESTRATOR_MODEL here too (rather than a hardcoded
# provider) means switching ORCHESTRATOR_MODEL to "ollama:..." moves every
# subagent onto Ollama in one place, not two. Set DATASET_AGENT_MODEL
# explicitly only if dataset-agent specifically needs a different model than
# everything else. It's tool-call-heavy (staging each source then one
# merge_and_split_dataset call) so, like sourcing-agent, it's kept on the
# same size model as the orchestrator by default rather than trimmed down.

# The Roboflow MCP server exposes 100+ tools (device management, workflows,
# vision-events, model training, etc.) - binding all of them blew the prompt
# to 26k+ tokens and tripped Groq's free-tier 8000 TPM limit on a single
# request. Trimming to the 9 tools plausibly needed to fetch/export a project
# still landed at 8022 tokens - 22 over the limit - so this is cut further to
# just the core fetch chain (find project -> fork -> generate a version ->
# export -> download_and_extract the result). Add back projects_get/
# projects_health/universe_dataset_images_search/image_upload individually if
# a real run turns out to need one of them; this allowlist is deliberately
# the minimum first guess, not a final answer - re-tune it once dataset-agent
# has actually run against real Roboflow projects a few times.
#
# `async_tasks_get` was missing here originally and is NOT optional:
# `projects_fork` is an async operation that only returns a `taskId` - the
# fork isn't actually done until polling `async_tasks_get(task_id=...)`
# returns a terminal status ("completed"/"failed"). Without this tool bound,
# dataset-agent has no way to ever confirm a fork finished, which is exactly
# what happened on a real live run: it found real Roboflow Universe sources
# but reported it couldn't stage them, because the fetch chain was missing
# a required step.
_ROBOFLOW_TOOL_ALLOWLIST = {
    "universe_search",
    "projects_fork",
    "async_tasks_get",
    "versions_generate",
    "versions_get",
    "versions_export",
}

# Kaggle's real MCP server (https://www.kaggle.com/mcp) exposes ~70 tools
# across its whole product surface - same bloat risk as Roboflow's 107.
# dataset-agent actually fetches (unlike sourcing-agent, which only
# catalogs - see its own, narrower allowlist in subagents/sourcing.py), so
# this keeps download_dataset/list_dataset_files in addition to the
# discovery tools.
_KAGGLE_TOOL_ALLOWLIST = {
    "get_dataset_info",
    "get_dataset_files_summary",
    "list_dataset_files",
    "download_dataset",
}


def _filter_roboflow_tools(roboflow_tools: list) -> list:
    return [t for t in roboflow_tools if getattr(t, "name", None) in _ROBOFLOW_TOOL_ALLOWLIST]


def _filter_kaggle_tools(kaggle_tools: list) -> list:
    return [t for t in kaggle_tools if getattr(t, "name", None) in _KAGGLE_TOOL_ALLOWLIST]


def build_dataset_agent(roboflow_tools: list, kaggle_tools: list) -> SubAgent:
    spec: SubAgent = {
        "name": "dataset-agent",
        "description": (
            "Fetches every mergeable sourced dataset (per sources.json), merges them, "
            "dedupes near-identical images across all of them, splits train/val/test, and "
            "emits a YOLO-format data.yaml. Call this after sourcing-agent has a sources.json "
            "you're satisfied with."
        ),
        "system_prompt": (
            "sources.json is written by sourcing-agent as a JSON array, one object per "
            "candidate source, shaped like:\n"
            '  {"source": "roboflow_universe", "dataset_id": "...", "url": "...", '
            '"classes_covered": ["car"], "image_count": 1200, "annotation_format": "YOLO", '
            '"license": "...", "annotation_coverage": {"car": 800}, "quality_notes": "...", '
            '"status": "available"}\n'
            "This run merges EVERY qualifying source, not just the biggest one - "
            "sources.json is a catalog of candidates, and more qualifying sources means a "
            "bigger merged dataset. Call list_qualifying_sources first: it filters to "
            "entries with status == \"available\" and annotation_format == \"YOLO\" (the "
            "only ones auto-mergeable right now) and returns every one of them (index, "
            "dataset_id, url, image_count). Fetch/stage EVERY index it lists - do not skip "
            "any qualifying entry. If list_qualifying_sources reports nothing qualifies "
            "(e.g. every source still needs annotation), say so plainly rather than fetching "
            "anything - annotation-agent is not part of the active roster in this run, so a "
            "\"needs_annotation\" source cannot be used no matter its image_count.\n\n"
            "Stage each listed source's raw files locally before merging: "
            "/workspace/sourced/<index>/images/, /workspace/sourced/<index>/labels/ "
            "(YOLO .txt, one per image) and /workspace/sourced/<index>/classes.txt (that "
            "source's own class names, one per line, in the numeric-ID order its label "
            "files use - required so class IDs can be safely remapped into one canonical "
            "list; without it, merging would silently scramble labels; a Roboflow YOLOv8 "
            "export's data.yaml `names:` field works too - merge_and_split_dataset reads "
            "either). <index> is the 0-based position list_qualifying_sources returned for "
            "that entry. Before fetching anything, check whether that index's files already "
            "exist at /workspace/sourced/<index>/ (e.g. ls it) - a prior run or a human may "
            "have already staged them there, and re-fetching is unnecessary work that can "
            "also fail for URLs that were never meant to be fetched directly (a Roboflow "
            "Universe project page, for instance, is a browsable URL, not a download link). "
            "Only fetch a listed source if its files are missing. Repeat the fetch for EVERY "
            "listed index before merging - one source's failure doesn't excuse skipping the "
            "others. For a Roboflow Universe source (found via universe_search, not one you "
            "already own), the fetch chain is exactly this sequence per source - none of "
            "these steps can be skipped or assumed:\n"
            "  1. projects_fork(url=<the source's Universe URL>) - this is ASYNC and only "
            "returns {taskId, url}, not a finished fork.\n"
            "  2. async_tasks_get(task_id=taskId) - poll every ~5s until status is "
            "'completed' (or 'failed', in which case report the failure and move on rather "
            "than retrying indefinitely).\n"
            "  3. versions_generate(project_id=...) using the forked project's id - omit "
            "preprocessing/augmentation (accept its defaults) since you have no way to "
            "confirm those choices with the user mid-run; note in your summary that "
            "defaults were used so the orchestrator can flag it if that matters.\n"
            "  4. versions_get(project_id, version_number) - poll until the version is "
            "ready, not still generating.\n"
            "  5. versions_export(project_id, version_number, export_format=\"yolov8\") - "
            "check/trigger the export; once it returns a download URL, pass that directly "
            "to download_and_extract(url=..., dest_dir=\"/workspace/sourced/<index>/\").\n"
            "For a Kaggle source, use download_dataset or a direct-download URL with "
            "download_and_extract instead - no fork/version chain applies there.\n"
            "If any step fails or a required tool isn't available, say so plainly and "
            "specifically in your summary (which step, what error) rather than treating "
            "the source as vaguely blocked or silently giving up.\n\n"
            "Once every listed source is staged, call merge_and_split_dataset once - call "
            "it with no path arguments (sources_json_path/sourced_dir/output_dir/"
            "class_budget_path) unless you have a real reason to override a default; its "
            "defaults (including writing to /workspace/dataset) are already correct and "
            "training-agent expects that exact path. It reads class_budget.json for the "
            "canonical class list (falls back to the union "
            "of every source's classes_covered if that file is missing), remaps each "
            "source's class IDs, drops near-duplicate images via perceptual hashing, splits "
            "by whole image into train/val/test (consult the cv-dataset-curation skill for "
            "the ratio/imbalance rules it applies), and writes dataset/data.yaml plus the "
            "underlying YOLO directory tree itself - you do not need to write those files "
            "by hand. Relay its returned summary as your final message, prefixed with which "
            "sources you skipped and why (mirroring the routing-signal style sourcing-agent "
            "uses: a short, plain-text summary the orchestrator can act on without re-reading "
            "any JSON file itself)."
        ),
        "tools": [
            *_filter_roboflow_tools(roboflow_tools),
            *_filter_kaggle_tools(kaggle_tools),
            list_qualifying_sources,
            download_and_extract,
            merge_and_split_dataset,
        ],
    }

    model = build_model(os.environ.get("DATASET_AGENT_MODEL"))
    if model is not None:
        spec["model"] = model

    return spec
