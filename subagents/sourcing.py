import os

from deepagents import SubAgent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from tools.model_builder import build_model
from tools.sourcing_builder import append_sources

_WRITE_FILE_BLOCK_MESSAGE = (
    "write_file is blocked for sources.json - call append_sources instead, "
    "passing the same data as its new_sources argument (a list of dicts, one "
    "entry per call). append_sources reads the existing file and updates-or-"
    "appends by (source, dataset_id) automatically - you do not need to "
    "construct the file's full contents yourself."
)


class _BlockWriteFileForSourcesJson(AgentMiddleware):
    """Short-circuits write_file calls targeting sources.json instead of
    letting them reach the real tool.

    Live runs have repeatedly shown this model ignoring the system prompt's
    explicit "never use write_file for sources.json" instruction and instead
    burning 8-10 consecutive turns hand-rolling write_file with a dict/list
    content (which the tool rejects, since content must be a plain string) -
    it either never gives up and stalls the whole run, or eventually gives up
    and falsely claims sources.json exists with the right content when it
    was never written at all. A system-prompt-only fix wasn't reliable
    enough on its own, so this intercepts the call in code and returns an
    unambiguous redirect instead of the tool's generic validation error,
    which the model has shown it doesn't reliably act on.
    """

    def _maybe_block(self, request):
        call = request.tool_call
        if call.get("name") != "write_file":
            return None
        file_path = str((call.get("args") or {}).get("file_path", ""))
        if "sources.json" not in file_path:
            return None
        return ToolMessage(content=_WRITE_FILE_BLOCK_MESSAGE, tool_call_id=call.get("id", ""))

    def wrap_tool_call(self, request, handler):
        blocked = self._maybe_block(request)
        return blocked if blocked is not None else handler(request)

    async def awrap_tool_call(self, request, handler):
        blocked = self._maybe_block(request)
        return blocked if blocked is not None else await handler(request)

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
            "You have EXACTLY these search tools and nothing else: Roboflow "
            "universe_search, Kaggle search_datasets, and (if bound) a web search "
            "tool. You have NO way to reach KITTI, Cityscapes, Waymo Open Dataset, "
            "OpenImages, ImageNet, COCO, or any other named public dataset unless "
            "one of your bound tools actually returned it BY NAME in its result "
            "this turn - the general fame or existence of a dataset is not a "
            "substitute for a real tool result. You also have NO download/fetch "
            "capability at all - never claim to have 'compiled a repository', "
            "'downloaded' anything, or created any images/ directory; fetching is "
            "dataset-agent's job, not yours. If you catch yourself about to name a "
            "dataset, stop and check: did a tool call THIS turn literally return "
            "this name? If not, do not write it to append_sources and do not "
            "mention it in your summary - say plainly that you didn't find enough "
            "and what you'd need (e.g. a broader web search, or a different "
            "provider) instead of inventing a plausible-sounding source list. "
            "Fabricating sources is worse than reporting a thin result, since "
            "dataset-agent and the orchestrator both trust your output as ground "
            "truth without re-verifying it.\n\n"
            "Search for public datasets and annotated data matching the classes "
            "you were asked about (all of them, or a specific subset named in your "
            "task). Prefer already-annotated sources.\n\n"
            "You must NEVER call write_file for sources.json, under any circumstances, "
            "even as a fallback when something else is going wrong. write_file's "
            "content argument requires a plain string, not a list/dict - passing a "
            "Python object there will always fail with a type error, and manually "
            "JSON-encoding it yourself throws away append_sources' whole purpose: "
            "reading the existing file first and updating-or-appending by "
            "(source, dataset_id) so nothing is ever silently overwritten. If "
            "append_sources itself errors, read the error message and fix your "
            "arguments to IT - do not switch tools. And never end a turn by asking "
            "the user (or anyone else) to manually create or edit sources.json - "
            "you are the only thing in this pipeline that writes to it; if you "
            "cannot make append_sources succeed, say plainly what error you're "
            "hitting and stop, rather than asking someone else to do your job.\n\n"
            "For every candidate you find, call append_sources separately for each "
            "one - pass a single-item list (`new_sources=[{...}]`), not a batch of "
            "several entries in one call. It reads sources.json first and updates-"
            "or-appends by (source, dataset_id), so calling it repeatedly is just as "
            "safe as one big call, and you never need to read the file yourself or "
            "worry about overwriting what a previous round already found. This "
            "matters specifically on this model: each entry is a nested object "
            "(annotation_coverage is itself a dict, classes_covered a list) and "
            "batching many of these into one tool-call argument has produced "
            "malformed JSON that fails outright - one entry per call is far more "
            "reliable. Don't pass sources_json_path unless you have a real reason to "
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
        "middleware": [_BlockWriteFileForSourcesJson()],
    }
    model = build_model(os.environ.get("SOURCING_AGENT_MODEL"))
    if model is not None:
        spec["model"] = model
    return spec
