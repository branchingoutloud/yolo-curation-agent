"""dataset-agent: fetches sourced data and merges/dedupes/splits it into a
YOLO-format dataset. Runs directly after sourcing-agent - annotation-agent is
temporarily excluded from the active roster (see agent.py), so any
sources.json entry with `status != "available"` is left pending rather than
routed anywhere; dataset-agent just reports it back to the orchestrator.
"""

import os

from deepagents import SubAgent

from tools.dataset_builder import download_and_extract, merge_and_split_dataset
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
            "Fetches sourced data (per sources.json), merges it, dedupes near-identical "
            "images, splits train/val/test, and emits a YOLO-format data.yaml. Call this "
            "after sourcing-agent has a sources.json you're satisfied with."
        ),
        "system_prompt": (
            "You have REAL, working tools right now, bound to this exact turn: Roboflow "
            "MCP tools (search/fork/version/export), Kaggle MCP tools, download_and_extract, "
            "and merge_and_split_dataset. This is a real, connected environment - these tool "
            "calls make real network requests and real filesystem writes against the actual "
            "workspace, right now, in this conversation. You must NEVER say you lack internet "
            "access, lack file-system manipulation ability, or that downloads 'cannot be "
            "performed in this environment' - all of that is false here, and saying it just "
            "wastes time by asking someone else to run a script that was never part of how "
            "this pipeline works. If your task involves fetching or merging data, the ONLY "
            "acceptable response is to actually call your bound tools THIS turn - never a "
            "bash script, git clone, curl, jq, kaggle CLI command, or any other shell command "
            "for a human to run elsewhere. You have no execute()/shell tool at all, and none "
            "of those commands are how this pipeline fetches data anyway (Roboflow Universe "
            "project pages aren't git repositories - git clone against one would never work). "
            "If you're unsure whether a tool call will succeed, the correct move is to try it "
            "and see what it actually returns, not to assume failure and describe a plan "
            "instead.\n\n"
            "Read sources.json first - it is written by sourcing-agent as a JSON array, "
            "one object per candidate source, shaped like:\n"
            '  {"source": "roboflow_universe", "dataset_id": "...", "url": "...", '
            '"classes_covered": ["car"], "image_count": 1200, "annotation_format": "YOLO", '
            '"license": "...", "annotation_coverage": {"car": 800}, "quality_notes": "...", '
            '"status": "available"}\n'
            "Only entries with status == \"available\" are mergeable right now - and only if "
            "they're actually annotated with real bounding boxes (YOLO .txt OR Pascal VOC "
            ".xml - merge_and_split_dataset auto-detects which from what's actually staged, "
            "not from this field's text, since it's been inconsistent in practice: values "
            "like \"bounding boxes, class labels\" or \"YOLO (Roboflow)\" both mean real, "
            "usable box annotations even though neither literally says \"YOLO\"). Sources "
            "annotated with something merge_and_split_dataset can't use yet (keypoints-only, "
            "segmentation masks, classification-only labels) or status == \"needs_annotation\" "
            "must NOT be merged or fabricated - annotation-agent is not part of the active "
            "roster in this run, so just leave those pending and name them in your summary "
            "(e.g. \"'bus' still needs annotation - 300 raw images found, 0 annotated\"). Do "
            "not invent labels for them.\n\n"
            "For each mergeable source, stage its raw files locally before merging: "
            "/workspace/sourced/<index>/images/ and /workspace/sourced/<index>/labels/. "
            "If the source's real annotations are YOLO .txt, also stage "
            "/workspace/sourced/<index>/classes.txt (that source's own class names, one per "
            "line, in the numeric-ID order its label files use - required so class IDs can "
            "be safely remapped into one canonical list; without it, merging would silently "
            "scramble labels). If the source's real annotations are Pascal VOC .xml instead "
            "(common for academic datasets like Stanford Dogs - check the actual annotation "
            "files, not just this field's text), stage the .xml files in labels/ directly - "
            "no classes.txt needed, VOC names each object's class as a string already. "
            "<index> is that entry's 0-based position in the sources.json array. Before "
            "fetching anything, "
            "check whether that index's files already exist at /workspace/sourced/<index>/ "
            "(e.g. ls it) - a prior run or a human may have already staged them there, and "
            "re-fetching is unnecessary work that can also fail for URLs that were never "
            "meant to be fetched directly (a Roboflow Universe project page, for instance, "
            "is a browsable URL, not a download link). Only if a source's files are missing "
            "should you fetch them. For a Roboflow Universe source (found via "
            "universe_search, not one you already own), the fetch chain is exactly this "
            "sequence - none of these steps can be skipped or assumed:\n"
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
            "IMPORTANT: the ONLY way /workspace/dataset/ (data.yaml, images/, labels/) comes "
            "into existence is by actually calling merge_and_split_dataset and getting its "
            "real returned summary back - you have no ability to write that directory tree "
            "or data.yaml by hand, and must never describe a merged dataset's file layout, "
            "image counts, or data.yaml content unless you already called that tool THIS turn "
            "and are relaying what it actually returned. If you have not made that call yet, "
            "say so plainly rather than presenting a plausible-looking directory listing.\n\n"
            "Once every mergeable source is staged, call merge_and_split_dataset once - call "
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
            "by hand. If your task asks for a specific total image count (e.g. \"select 30 "
            "of the 100 available\"), pass that number as merge_and_split_dataset's "
            "max_total_images argument - it will randomly subset AFTER dedup and still "
            "produce a real, correctly-split dataset. You have NO execute()/shell tool at "
            "all - you cannot run mkdir/cp/for-loops, and neither can the user run a script "
            "you write out for them as a substitute for actually calling your own tool. "
            "Never respond with shell commands, a manual copy script, or instructions telling "
            "someone else to create the directories - if merge_and_split_dataset can't do "
            "something, say so plainly and stop, don't invent a workaround you can't execute. "
            "Relay its returned summary as your final message, prefixed with which "
            "sources you skipped and why (mirroring the routing-signal style sourcing-agent "
            "uses: a short, plain-text summary the orchestrator can act on without re-reading "
            "any JSON file itself)."
        ),
        "tools": [
            *_filter_roboflow_tools(roboflow_tools),
            *_filter_kaggle_tools(kaggle_tools),
            download_and_extract,
            merge_and_split_dataset,
        ],
    }

    model = build_model(os.environ.get("DATASET_AGENT_MODEL"))
    if model is not None:
        spec["model"] = model

    return spec
