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
# export -> download_and_extract the result). Add back projects_health/
# universe_dataset_images_search/image_upload individually if a real run
# turns out to need one of them; this allowlist is deliberately the minimum
# first guess, not a final answer - re-tune it once dataset-agent has
# actually run against real Roboflow projects a few times. (The free-tier
# token-limit problem itself is moot now that model calls run on Ollama
# Cloud - no such per-request ceiling - but the allowlist is kept trim
# anyway since it's still good practice.)
#
# `async_tasks_get` was missing here originally and is NOT optional:
# `projects_fork` is an async operation that only returns a `taskId` - the
# fork isn't actually done until polling `async_tasks_get(task_id=...)`
# returns a terminal status ("completed"/"failed"). Without this tool bound,
# dataset-agent has no way to ever confirm a fork finished, which is exactly
# what happened on a real live run: it found real Roboflow Universe sources
# but reported it couldn't stage them, because the fetch chain was missing
# a required step.
#
# `projects_get` and `projects_list` were added so dataset-agent can check
# whether a Universe source has already been forked (and already has a
# version) before forking or generating again - previously projects_fork/
# versions_generate ran unconditionally every single time dataset-agent
# processed the same source, silently creating a brand-new duplicate
# fork/version on every re-run (e.g. a later iteration round after eval-agent
# flags a weak class).
#
# IMPORTANT, confirmed by direct testing against the real MCP server:
# `projects_get(project_id=<original Universe slug>)` succeeds and returns
# full project details (images, classes, version history) EVEN IF the
# caller has never forked that project - Roboflow Universe project metadata
# is public. A successful projects_get on the source's own slug is NOT
# proof of an existing fork, only that the project is public. Forked copies
# also get a new, randomized project_id/slug (e.g. a Universe project named
# "kangaroo" forks to something like "kangaroo-rkb3c-dgtlw" in the caller's
# own workspace) - guessing that a fork reuses the source's original slug
# does not reliably work either. The only reliable source of truth for
# "have I already forked this" is `projects_list`, which is scoped to the
# credential's own workspace - see the system_prompt for the exact check
# (list once per run, match by name, not by assumed project_id).
#
# Also confirmed directly: versions_export's real response shape includes
# `ready` (bool) and `progress` (0-100) - e.g. `{"ready": false, "progress":
# 0}` for a freshly-triggered real export, no `link` field at all until
# ready. A response that jumps straight to a `link` for a project that was
# never actually verified-forked (see above) can produce a link that 404s
# forever on download - that combination is a sign the project/version
# isn't genuinely owned, not a "still processing, retry" situation.
_ROBOFLOW_TOOL_ALLOWLIST = {
    "universe_search",
    "projects_fork",
    "projects_get",
    "projects_list",
    "async_tasks_get",
    "versions_generate",
    "versions_get",
    "versions_export",
}

# dataset-agent no longer binds any Kaggle tools: list_qualifying_sources
# (tools/dataset_builder.py) now hard-filters to Roboflow Universe sources
# only (checked against the URL's actual domain, not the free-text `source`
# field), so a Kaggle entry can never be selected for fetching - Kaggle
# tools would be dead weight. sourcing-agent still searches/catalogs Kaggle
# (see subagents/sourcing.py) for visibility; it's just never fetched here.
# `kaggle_tools` is kept as a parameter below purely so existing call sites
# (agent.py, scripts/*) don't need to change - it's intentionally unused.


def _filter_roboflow_tools(roboflow_tools: list) -> list:
    return [t for t in roboflow_tools if getattr(t, "name", None) in _ROBOFLOW_TOOL_ALLOWLIST]


def build_dataset_agent(roboflow_tools: list, kaggle_tools: list) -> SubAgent:  # noqa: ARG001 - kaggle_tools kept for call-site compatibility, see comment above
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
            "This run merges up to 3 qualifying Roboflow Universe sources (the largest, by "
            "image_count), not just the single biggest one - but capped, not unlimited: "
            "forking and exporting every qualifying source has real Roboflow API cost (a "
            "fork + version-generate + export each) for shrinking benefit once the largest "
            "few are already merged. Only Roboflow Universe sources are auto-fetchable "
            "right now - dataset-agent's fetch chain only knows how to fork/export from "
            "Roboflow, so any other source (e.g. Kaggle) that otherwise qualifies is never "
            "selected, no matter how large. Call list_qualifying_sources first: it filters "
            "to entries with status == \"available\", annotation_format == \"YOLO\", and a "
            "url actually on roboflow.com, already applies both that restriction and the "
            "cap in code, and returns exactly the sources you should fetch (index, "
            "dataset_id, url, image_count) plus any other qualifying sources it "
            "deliberately excluded (for being non-Roboflow, or for being past the cap). "
            "Fetch/stage EVERY index it lists - do not skip any of them - and do not fetch "
            "anything beyond what it returns, even if sources.json has more qualifying "
            "entries; mention any excluded sources in your final summary so the "
            "orchestrator knows more data exists if a future round wants it. If "
            "list_qualifying_sources reports nothing qualifies "
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
            "already own), first check whether it's already been forked into your own "
            "workspace before creating anything new - forking again or generating another "
            "version of a project you already forked just creates wasteful duplicates. "
            "IMPORTANT: projects_get(project_id=<the source's original Universe slug>) is "
            "NOT a valid way to check this - Universe project metadata is public, so "
            "projects_get succeeds and returns real project/version data for ANY public "
            "Universe project whether or not you have ever forked it. A forked copy also "
            "gets a brand-new, randomized project_id (e.g. a Universe project \"kangaroo\" "
            "forks to something like \"kangaroo-rkb3c-dgtlw\" in your own workspace), so "
            "guessing the fork reuses the source's original slug will not reliably find "
            "it either. The only reliable check is projects_list, which is scoped to your "
            "own workspace:\n"
            "  1. Call projects_list() once per dataset-agent run (not once per source) "
            "and keep the result - it returns every project already in your own workspace "
            "as {projects, total, limit, offset}.\n"
            "  2. For each source, check whether any entry in that list matches it by "
            "name (case-insensitive, tolerant of a randomized slug suffix - compare "
            "against the `name` field, not an assumed project_id). If found, use that "
            "entry's real `id` as project_id for every step below and skip straight to "
            "step 5 - do not fork it again.\n"
            "  3. projects_fork(url=<the source's Universe URL>) - only if step 2 found "
            "no match. This is ASYNC and only returns {taskId, url}, not a finished fork.\n"
            "  4. async_tasks_get(task_id=taskId) - poll every ~10s until status is "
            "'completed' (or 'failed', in which case report the failure and move on rather "
            "than retrying indefinitely). Once completed, this response gives you the "
            "forked project's real id - use that, never the original Universe slug.\n"
            "  5. projects_get(project_id=<the real id from step 2 or step 4>) - inspect "
            "its `versions` list. If at least one version already exists, reuse its "
            "version_number and skip straight to step 7 - do NOT call versions_generate "
            "on a project that already has a version, since that creates a new duplicate "
            "version every time this runs.\n"
            "  6. versions_generate(project_id=...) - only if step 5 found no existing "
            "version. Omit preprocessing/augmentation (accept its defaults) since you have "
            "no way to confirm those choices with the user mid-run; note in your summary "
            "that defaults were used so the orchestrator can flag it if that matters.\n"
            "  7. versions_get(project_id, version_number) - poll every ~10s until the "
            "version is ready, not still generating.\n"
            "  8. versions_export(project_id, version_number, export_format=\"yolov8\") - "
            "the response includes `ready` (bool) and `progress` (0-100), e.g. "
            "`{\"ready\": false, \"progress\": 0}` for a freshly-triggered real export, "
            "with no `link` field at all until ready. While `ready` is false, wait ~10s "
            "and call versions_export again rather than treating an in-progress response "
            "as a failure - keep polling this way for up to ~2 minutes total; larger "
            "datasets may genuinely need longer, so if it is still not ready after that, "
            "say so explicitly in your summary rather than silently giving up. Only once "
            "`ready` is true and a real download link is present, pass that link directly "
            "to download_and_extract(url=..., dest_dir=\"/workspace/sourced/<index>/\"). "
            "If download_and_extract ever fails on a link that WAS reported ready (e.g. a "
            "404 on the underlying storage URL) - that is NOT a \"still processing, keep "
            "retrying\" situation, it is a sign this project/version was never genuinely "
            "yours to export (most likely step 2's name-match was wrong and this is "
            "actually still someone else's public Universe project). Do not keep retrying "
            "blindly in that case - report the exact error and which project_id was used, "
            "so the mismatch can be investigated rather than masked by more retries.\n"
            "Forking, generating a version, and exporting only stage this source's raw "
            "files locally under /workspace/sourced/<index>/ - that is not the final "
            "dataset. This source still goes through the same merge_and_split_dataset step "
            "as every other staged source below (pooled, deduped by perceptual hash, "
            "class-remapped, and re-split); nothing about a Roboflow fork is special-cased "
            "there. Nothing in this chain ever uploads the merged result back to Roboflow "
            "either - the merged dataset only ever exists locally under output_dir, never "
            "on the Roboflow dashboard, so don't expect to see it there.\n"
            "list_qualifying_sources only ever returns Roboflow Universe sources - "
            "anything else (e.g. Kaggle) that otherwise qualifies is reported separately "
            "as excluded, not selected for fetching, since only the Roboflow fork/export "
            "chain above is auto-fetchable right now. Mention any such excluded sources "
            "in your summary too, the same way you'd mention cap-excluded ones.\n"
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
            list_qualifying_sources,
            download_and_extract,
            merge_and_split_dataset,
        ],
    }

    model = build_model(os.environ.get("DATASET_AGENT_MODEL"))
    if model is not None:
        spec["model"] = model

    return spec
