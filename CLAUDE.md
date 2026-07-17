# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A `deepagents`-native orchestrator agent that takes a user's object-detection use case,
sources and curates training data, trains a YOLO model, evaluates it, and proposes
iterations — autonomously, with human approval at three checkpoints. Design rationale
lives in `yolo-deep-agent-architecture.md`, which is **not checked into this repo**
(referenced by section number in code comments, e.g. "§9 of the architecture doc" —
those section numbers won't resolve to anything locally).

The UI is a separate sibling repo (`../deep-agents-ui`, Node/Next.js) run independently
and pointed at this agent's LangGraph server.

## Setup & running

```bash
uv sync                                   # creates/updates .venv from pyproject.toml + uv.lock
cp .env.example .env                      # fill in ANTHROPIC_API_KEY at minimum
uv run langgraph dev                      # serves at http://127.0.0.1:2024
```

Dependencies are declared in `pyproject.toml`; `uv.lock` pins the full resolved
set and should be committed. Adding a new dependency means editing
`pyproject.toml`'s `dependencies` list, then re-running `uv sync` - don't
`pip install` a package directly into `.venv`, since `uv sync` will treat
anything not declared in `pyproject.toml` as drift and remove it on the next
sync (this bit us when `langchain-groq` was pip-installed ad hoc, then
disappeared on the next `uv sync` until it was added to `dependencies`).

Sanity check the server booted without a UI:

```bash
curl http://127.0.0.1:2024/assistants/search -X POST -H "Content-Type: application/json" -d '{}'
```
Should return an assistant with ID `yolo-agent` (see `langgraph.json`, which maps that
graph key to `./agent.py:agent`).

There is no lint/test/build tooling configured in this repo (no pytest, no linter config,
no CI) — verification is currently "does `langgraph dev` boot and does the graph run."

### Env vars are optional-by-degradation, not optional-by-absence

Only `ANTHROPIC_API_KEY` is required to boot. `ROBOFLOW_API_KEY`, `KAGGLE_MCP_URL`,
`TAVILY_API_KEY`, and `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` are each independently
optional: missing one just shrinks that subagent's tool list to `[]` at startup
(see `tools/mcp_clients.py`) rather than raising. Never make an integration hard-required
at import time — follow the existing lazy/defensive pattern (try, warn, return `[]`).

### Model wiring (`tools/model_builder.py`) — Ollama Cloud, per-subagent overrides

Every model spec in this repo (`ORCHESTRATOR_MODEL` and each subagent's optional
`*_AGENT_MODEL` override) goes through `tools/model_builder.py`'s `build_model()`
instead of being passed as a raw string to `create_deep_agent`/`init_chat_model`
directly — this is what lets an `"ollama:..."` spec pick up Ollama Cloud's
base_url/Bearer-auth wiring. If you add a new subagent or a new standalone test
script, route its model through `build_model()` too rather than hand-rolling the
provider string, or it'll silently fall back to local/unauthenticated Ollama
defaults and fail against a cloud key.

Currently running on Ollama Cloud (`https://ollama.com`), authenticated via
`OLLAMA_API_KEY`. Per-subagent overrides (`PLANNING_AGENT_MODEL`,
`SOURCING_AGENT_MODEL`, `DATASET_AGENT_MODEL`, `TRAINING_AGENT_MODEL`,
`EVAL_AGENT_MODEL` — all optional, all fall back to `ORCHESTRATOR_MODEL` if unset,
same `spec.get("model", model)` mechanism as before) split load by how
tool-call-heavy each subagent's job is: `gpt-oss:120b` for orchestrator/sourcing/
dataset (heavy multi-step tool use), `gpt-oss:20b` for training/eval (more
mechanical, and currently non-functional anyway per Known stubs). Confirmed via a
real `/api/tags` + `/api/chat` call against this account (2026-07-17): cloud model
tags need **no** `-cloud` suffix — the catalog lists them bare. Catalog changes
over time; re-check `GET {OLLAMA_BASE_URL}/api/tags` with your own key rather than
trusting this list to stay current.

**This is a free-tier key: only ONE cloud call may be in flight at a time** — a
second concurrent call errors instead of queueing. `build_model()` wraps every
cloud model in `_ThrottledCloudChatOllama`, which serializes all four
invoke/stream/ainvoke/astream entry points behind a process-wide lock (separate
sync/async locks — see the class docstring for why that split is an acceptable
approximation here). Don't remove this lock while still on a free-tier key; the
orchestrator is allowed to issue parallel `task` calls in one turn, and without
the lock two subagents hitting the API at once will both fail. `scripts/
check_ollama.py` is the preflight check — confirms the key/base_url reach the API
and that the configured model actually responds, before spending several agent
turns on a connection/auth/404 that would otherwise only surface mid-run.

`_ThrottledCloudChatOllama` also retries transient `ollama._types.ResponseError`
5xx responses (up to 3 attempts, short backoff) — observed live, twice, on this
free-tier key with an otherwise-valid request; an uncaught exception from the
model node crashes the whole graph exactly like an uncaught exception from a
`@tool` does (see the dataset-agent path-argument story below), so one transient
blip used to kill an entire orchestrator run through gate 2. 4xx errors (bad
model name, bad auth) are deliberately NOT retried — those never succeed on
retry. Streaming retries only kick in if nothing has been yielded yet in that
attempt, to avoid duplicating output on a genuine mid-stream failure.

HuggingFace is a documented but unwired fallback: any `*_AGENT_MODEL` can be
pointed at `"huggingface:<repo_id>"` the moment `langchain-huggingface` is added
to `pyproject.toml` and a real `HUGGINGFACEHUB_API_TOKEN` exists — useful if the
single-concurrency cloud key becomes a bottleneck under real load, since it would
let one subagent's calls run on a separate provider instead of queuing behind
everyone else's. Not wired to any subagent today; no token/model has been chosen.

## Architecture

**Orchestrator + subagents, all built with `deepagents.create_deep_agent`/`SubAgent`.**
`agent.py` is the only wiring point: it builds the active subagents, loads MCP/tool lists,
and constructs the orchestrator with a shared backend. `annotation-agent` is currently
excluded from the active roster (commented-out imports/wiring in `agent.py`, code intact
in `subagents/annotation.py`) — sourcing-agent hands off directly to dataset-agent, which
treats any `sources.json` entry with `status != "available"` as pending rather than
assuming annotation-agent will pick it up. The orchestrator itself has *no*
data/annotation/training tools — its only non-delegation tool is `request_approval`
(`tools/approval.py`). Everything else happens via `task` delegation to subagents, and
the orchestrator's job is to plan (`write_todos`), delegate, read subagent output files
back off the filesystem, and decide whether another iteration round is warranted.

### The three approval gates

`request_approval` is a dedicated tool (not an interrupt on `task` itself — an interrupt
on every subagent call would fire far too often). It must be called at exactly three
points, per `agent.py`'s `ORCHESTRATOR_PROMPT`:
1. after a sourcing/image-budget plan, before any data is pulled,
2. after a proposed YOLO model size, before training starts,
3. after eval results, before proposing another sourcing/annotation/training round.

If you change orchestrator behavior, preserve this three-gate contract — it's enforced
by prompt instruction, not code, so it's easy to accidentally erode when editing
`ORCHESTRATOR_PROMPT`.

### Subagent pipeline (`subagents/`)

Each subagent is a plain dict factory function returning a `SubAgent`. They form a rough
pipeline but the orchestrator can re-invoke any of them out of order based on eval results.
Active roster (`agent.py`'s `subagents=[...]` list): planning, sourcing, dataset, training,
eval. `annotation-agent` exists in the codebase but is not currently wired in — see above.

- **planning-agent** — estimates images-per-class from class variability, then calls the
  custom `write_plan` tool (`tools/planning_builder.py`) once with its full reasoning
  (per-class variability tier + images_per_class + rationale) — real Python, not a
  prompted write_file call. `write_plan` checks each class's count against the
  cv-dataset-curation skill's floor for its stated tier and flags "thin" classes (under
  30% of the median), then writes both `plan.md` and `class_budget.json` itself
  (`{"classes": [...], "budgets": {...}, "variability": {...}, "split_ratios": {...}}` —
  the shape dataset-agent's `_canonical_class_list` already expects). Verified end-to-end
  with a real model call (see Testing below) — correctly picked tiers at/above the
  skill's floors with no warnings on a normal case.
- **sourcing-agent** — searches Roboflow Universe / Kaggle / web for existing datasets;
  callable repeatedly for narrower follow-up gap-fills. Calls the custom `append_sources`
  tool (`tools/sourcing_builder.py`) instead of hand-writing JSON via write_file — it reads
  `sources.json` first (if present) and updates-or-appends by `(source, dataset_id)` key,
  code-enforcing the "never overwrite" contract rather than relying on the model to
  remember it every call, and rejects (rather than silently writing) any entry missing
  `source`/`dataset_id`/`status`. Entry shape: `{source, dataset_id, url, classes_covered,
  image_count, annotation_format, license, annotation_coverage, quality_notes, status}`.
  Its final message relays `append_sources`' summary plus its own read on what's still
  thin/unlabeled (e.g. "'bus' unlabeled — 300 raw images on Kaggle") — that IS the routing
  signal downstream agents act on, not something they re-derive by re-reading the JSON, so
  its system_prompt explicitly forbids mentioning any source it didn't actually pass to
  `append_sources` that same turn (added after a live Ollama Cloud run described a
  "HuggingFace" result in prose that was never in `sources.json` — sourcing-agent has no
  HuggingFace search tool bound at all, so that was pure fabrication in the summary, not a
  real find; the JSON file itself is always the ground truth to check against).
  Verified directly (bypassing the LLM): add/update/reject-malformed all behave correctly
  (see Testing below). **Now also verified live end-to-end** against real Roboflow Universe
  search on Ollama Cloud (`gpt-oss:120b`) — a real multi-tool-call search-then-catalog run
  that never once completed on Groq/Gemini free tiers (see Testing below).
- **annotation-agent** *(not in the active roster)* — zero-shot pre-labels classes with
  no annotated data found; writes `annotation_manifest.json`. `tools/zero_shot_annotate.py`
  is unimplemented regardless (see Known stubs).
- **dataset-agent** — for each `sources.json` entry with `status == "available"` and
  `annotation_format == "YOLO"`, fetches its raw files (Roboflow/Kaggle MCP tools, or the
  custom `download_and_extract` tool for a direct URL) into
  `/workspace/sourced/<index>/{images/,labels/,classes.txt}`, then calls the custom
  `merge_and_split_dataset` tool (`tools/dataset_builder.py`) — real Python, not just a
  prompted `execute()` call — which remaps each source's YOLO class IDs through its own
  `classes.txt` into one canonical list (from `class_budget.json`, else the union of every
  source's `classes_covered`), drops near-duplicates via perceptual hashing
  (`imagehash.phash`, default Hamming threshold 5), splits by whole image into
  train/val/test per the cv-dataset-curation skill's ratio rules, and writes
  `dataset/data.yaml` + the YOLO directory tree itself. Entries with any other `status`
  (e.g. `needs_annotation`) are left pending and named in the returned summary, never
  merged or fabricated. `DATASET_AGENT_MODEL` is an **optional** `"provider:model"`
  override (`subagents/dataset.py`, resolved via `tools/model_builder.py` — see Model
  wiring above) — if unset, dataset-agent inherits `ORCHESTRATOR_MODEL` automatically,
  identically to planning/sourcing/training/eval-agent (deepagents' subagent-building loop
  does `spec.get("model", model)`); only set it if dataset-agent specifically needs a
  different model than everything else. Both Roboflow's and Kaggle's MCP servers expose
  100+/~70 tools respectively (device management, competitions, notebooks, forums, etc.) —
  binding all of them blew a single request past every free-tier token limit tried on Groq/
  Gemini (see git history if chasing that era's exact numbers). `_ROBOFLOW_TOOL_ALLOWLIST`/
  `_KAGGLE_TOOL_ALLOWLIST` in both `subagents/dataset.py` and `subagents/sourcing.py` still
  trim each to a handful of relevant tools (good practice regardless of provider), but the
  free-tier-token-limit problem itself is moot now that both subagents run on Ollama Cloud
  (no such per-request token ceiling). **Live end-to-end runs now complete successfully** —
  see Testing below — including one that surfaced two real system_prompt gaps, both fixed:
  (1) dataset-agent wasn't told to check whether a source's files were already staged
  under `/workspace/sourced/<index>/` before fetching, so it tried (and failed) to
  `download_and_extract` a Roboflow *project page* URL instead of noticing the files were
  already there; (2) the model tended to explicitly pass path-override arguments
  (`sources_json_path`, `output_dir`, etc.) with self-invented values instead of omitting
  them and using the already-correct defaults — the prompts now say explicitly not to pass
  those unless overriding for a real reason. Separately, `tools/dataset_builder.py` and
  `tools/sourcing_builder.py`'s tool functions now catch `ValueError`/`FileNotFoundError`
  from bad path arguments and return a plain-text error instead of raising — an uncaught
  exception inside any `@tool` function crashes the whole graph run (LangGraph's default
  tool-error handling only catches `ToolException`, not arbitrary exceptions), which is
  exactly what happened live before this fix, once, on a self-invented bad path. The tool
  *logic* itself (merge/dedupe/remap/split, and append/update/reject) was separately
  verified correct by direct invocation before ever spending a model call on it (see
  Testing below) — don't conflate "the live run didn't complete" (no longer true) with "the
  code doesn't work" (was never true).
- **training-agent** — picks model size, runs `ultralytics` training via `execute()` in a
  real GPU sandbox (Modal); writes `runs/train/status.md` (progress) and
  `runs/train/metrics.json`. Built via `subagents/sandbox_subagent.py`'s
  `build_sandbox_subagent()` as its own independently-compiled agent with
  `training_sandbox_backend` as its backend — not a declarative `SubAgent`, since
  deepagents' shared-backend limitation (see Known stubs) makes that path inert for GPU
  work. `dataset/` and `model_choice.json` are copied into the sandbox before it runs;
  `runs/train/status.md`/`metrics.json` are copied back to real disk after (even on
  failure). Verified by inspection that its inner graph has a real `execute` tool bound
  — **not yet verified via an actual live GPU training run** (real Modal cost/time).
- **eval-agent** — reuses the training sandbox (same `ModalSandbox` singleton, so trained
  weights are already local there — no re-upload needed), builds a confusion matrix via
  `supervision`, diagnoses *why* each weak class underperforms (not just which metric is
  low), writes `eval_report.md` + `weak_classes.json` back to real disk. Deliberately does
  **not** decide the next action — that's the orchestrator's call. Same
  `build_sandbox_subagent()` construction and same "not yet live-verified" caveat as
  training-agent.

Subagents coordinate purely through files on the shared filesystem backend, not through
return values or shared state — always read the upstream JSON/markdown file rather than
assuming what a prior subagent produced.

### Skills (`skills/`)

Three skills provide the domain heuristics subagents are told to "consult" rather than
having the numbers hardcoded in prompts: `yolo-model-selection` (size vs. deployment
target, decision rule for stepping up a size), `cv-dataset-curation` (images-per-class
floors by variability tier, imbalance thresholds, split ratios, dedup checks),
`cv-eval-and-iteration` (mAP50 vs mAP50-95 interpretation, confusion-matrix-pattern →
likely-cause table). When adjusting these heuristics, edit the skill's `SKILL.md`
instead of a subagent's `system_prompt` — that's the intended single source of truth.

### Backends (`backends/`) — two different kinds

- **`project_backend.py`** — the shared virtual filesystem every subagent reads/writes
  through. `/workspace/*` is routed to real disk under `RUN_ARTIFACTS_DIR`
  (`./run_artifacts` by default) via `FilesystemBackend`; anything else falls back to
  ephemeral `StateBackend`. This is what makes `plan.md`, `sources.json`, `dataset/`,
  `runs/`, `eval_report.md` inspectable on disk after (or during) a run.
- **`sandboxes.py`** — `ModalSandbox`, a `BaseSandbox` subclass wrapping the `modal`
  Python SDK for `execute()`/`upload_files()`/`download_files()` (deepagents ships no
  built-in cloud sandbox with GPU support). Two long-lived singletons:
  `training_sandbox_backend` (GPU, `ultralytics` image, reused by eval-agent so trained
  weights are already local) and `annotation_sandbox_backend` (lighter, for zero-shot
  detector batches). Each lazily creates **one** Modal Sandbox container on first
  `execute()`/etc. call and reuses it for the rest of the process's life — it is not
  spun up per call. Don't change that to per-call sandbox creation without updating the
  `atexit` teardown logic too.

  **Now genuinely wired to training-agent/eval-agent** via
  `subagents/sandbox_subagent.py` (see Known stubs below for the full story) — the two
  subagents that need GPU compute are built as their own independent
  `create_deep_agent(backend=training_sandbox_backend, ...)` graphs instead of declarative
  `SubAgent` dicts, since deepagents==0.6.12 only ever applies a `SubAgent`'s `"backend"`
  key from the ORCHESTRATOR's shared `create_deep_agent` call, never a per-subagent one.
  `annotation_sandbox_backend` remains unused (`annotation-agent` isn't in the active
  roster).

`ultralytics`/`supervision` are intentionally **not** in `pyproject.toml` dependencies —
they only ever run inside the Modal sandbox image, never imported by the local
orchestrator process. Don't add them to local deps "for convenience"; it defeats the
point of keeping torch out of the orchestrator's venv.

## Testing (`scripts/`)

There's no pytest suite (see above) — `scripts/` has standalone harnesses for exercising
`planning-agent`/`sourcing-agent`/`dataset-agent` without needing the full orchestrator or
a real langgraph server. Every harness builds a standalone `create_deep_agent` (no
subagents of its own) directly from that subagent's `build_*_agent()`-returned spec — same
system_prompt/tools/model/backend a real `task()` delegation would give it, minus the
orchestrator/HITL-gate layer, so it's a faithful proxy for the real thing.

- `scripts/test_planning_agent.py` — dummy use case, no external network dependency
  (`write_plan` is pure local file writes). Verified working end-to-end with a real model
  (both on Groq originally and again on Ollama Cloud's `gpt-oss:120b`).
- `scripts/seed_dummy_workspace.py` / `scripts/test_dataset_agent.py` — synthetic images
  (random noise, not flat colors — phash needs real structure to tell "different" from
  "duplicate") pre-staged under an isolated `run_artifacts_test/`, so the merge/dedupe/
  remap/split logic runs with zero network dependency. Safe to rerun anytime; wipes and
  reseeds its workspace on every run. This is the one to use when iterating on
  `tools/dataset_builder.py` itself. **Verified live end-to-end on Ollama Cloud**
  (`gpt-oss:120b`) after the two system_prompt fixes described above: correctly notices
  pre-staged files, merges 4/5 images (drops exactly the one true near-duplicate), writes
  to `/workspace/dataset` with the right train/val split.
- `scripts/test_sourcing_agent.py` — a real (lightweight) Roboflow/Kaggle search.
  **Verified live end-to-end on Ollama Cloud** (`gpt-oss:120b`) — a real
  `universe_search`-then-`append_sources` run that wrote genuine Roboflow Universe
  entries to `sources.json`, something that never once completed on Groq/Gemini free
  tiers due to rate limits (see git history for that era's exact token-limit numbers).
- `scripts/seed_real_sources.py` / `scripts/test_dataset_agent_real.py` — a real
  sources.json (actual Roboflow Universe projects, nothing staged locally) against an
  isolated `run_artifacts_real_test/`, to test whether dataset-agent can discover and use
  its actual Roboflow MCP tools.
- `scripts/check_ollama.py` — preflight check for whichever `ORCHESTRATOR_MODEL`/
  `DATASET_AGENT_MODEL` is set to `"ollama:..."`: hits `/api/tags` (best-effort — may not
  reflect Ollama Cloud's hosted catalog) and then a real `/api/chat` call (the check that
  actually matters) before spending agent turns discovering a connection/auth/model-name
  problem mid-run. Works against both cloud (`OLLAMA_API_KEY` set) and local Ollama.
- Ollama Cloud calls can occasionally 500 transiently (observed once during this session,
  on an otherwise-correct request) — if a live test fails with `ollama._types.
  ResponseError: Internal Server Error`, just retry once before assuming something in the
  code regressed.
- Windows' default console codepage (cp1252) can't encode Unicode punctuation cloud models
  routinely emit (em-dashes, non-breaking hyphens) — every `scripts/test_*.py` now calls
  `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` near the top so printing a
  real model response doesn't crash the script with `UnicodeEncodeError`.
- **When a subagent's tool logic can be tested without an LLM at all (any custom
  `@tool`-decorated function), do that first** — `merge_and_split_dataset` and
  `append_sources` were both verified this way (call `.invoke({...})` directly against
  seeded files, assert on the result/output files) before ever spending a model call on
  them. This is how planning/sourcing/dataset-agent's core logic all got verified correct
  even though the *live full-agent* runs against Roboflow/Kaggle kept hitting free-tier
  limits — don't conflate "the live run didn't complete" with "the code doesn't work."
- **Use `ainvoke`, not `invoke`, in any such harness once real MCP tools are involved** —
  `langchain_mcp_adapters` tools are async-only (`StructuredTool` with only `_arun`
  implemented); sync `.invoke()` raises `NotImplementedError: StructuredTool does not
  support sync invocation` the moment the LLM calls one. This isn't a production concern
  (`langgraph dev` already serves the graph asynchronously) but bit these test scripts
  during development.
- **If a harness is itself `async def main()`**, fetch `get_roboflow_tools()`/
  `get_kaggle_tools()` *before* `asyncio.run(main(...))`, not inside it — both internally
  call their own `asyncio.run()`, which raises "cannot be called from a running event
  loop" (silently caught by `mcp_clients.py`'s broad `except Exception`, so it just looks
  like zero tools loaded rather than an obvious error) if called from within an
  already-running loop.
- **A subagent with no model override of its own needs an explicit `model=` when tested
  standalone** (`os.environ.get("ORCHESTRATOR_MODEL", ...)`, matching `agent.py`) —
  `create_deep_agent(model=None, ...)` falls back to deepagents' own built-in default
  (Anthropic), which fails outright without `ANTHROPIC_API_KEY` set. Bit both
  `test_planning_agent.py` and `test_sourcing_agent.py` until fixed — and route that
  spec through `tools/model_builder.py`'s `build_model()`, not a raw string, or an
  `"ollama:..."` spec silently skips the Ollama Cloud base_url/auth wiring and fails
  against localhost instead (bit both scripts a second time until fixed this way).

## Known stubs / incomplete pieces

Check these before assuming a code path is fully wired:

- **Per-subagent `"backend"` overrides do nothing in the installed `deepagents` version
  (0.6.12) — FIXED for training-agent/eval-agent via `subagents/sandbox_subagent.py`.**
  `SubAgent` (see `.venv/.../deepagents/middleware/subagents.py`) has no `backend` field,
  and `create_deep_agent`'s subagent-building loop (`graph.py`) always binds every
  declarative `SubAgent`'s `FilesystemMiddleware` to the single `backend=` passed to
  `create_deep_agent` itself — never anything from an individual subagent's spec dict. A
  plain `"backend": sandbox_backend` key in a `SubAgent` dict is silently ignored (Python
  dicts don't enforce `TypedDict` shape at runtime) — that's what `annotation.py` still
  does (moot; not in the active roster) and what `training.py`/`eval.py` used to do.
  **The real fix**: `subagents/sandbox_subagent.py`'s `build_sandbox_subagent()` builds
  training-agent/eval-agent as their own, separately-compiled
  `create_deep_agent(backend=sandbox_backend, ...)` graphs, then wraps each as a
  `CompiledSubAgent` (`{"name", "description", "runnable"}` — a different, less-common
  `deepagents` subagent shape than the declarative `SubAgent` dict every other subagent
  uses). Per `graph.py`'s subagent-building loop, `CompiledSubAgent` entries are used
  AS-IS, never rebuilt against the orchestrator's shared `backend=` — this is what lets
  them have a real, independent backend at all. Verified by direct inspection (not a live
  run): building `create_deep_agent(backend=training_sandbox_backend, ...)` and walking
  its compiled graph's `tools` node shows `execute` genuinely bound, alongside the usual
  `ls`/`read_file`/`write_file`/`edit_file`/`glob`/`grep` — confirms the wiring, not just
  the intent.
  The catch this introduces: a subagent with its OWN backend has filesystem tools that
  operate against THAT backend's filesystem, not `project_backend`'s real disk — so
  whatever the orchestrator/dataset-agent already wrote (`dataset/`, `model_choice.json`)
  is invisible to it unless copied in first, and whatever it writes back
  (`runs/train/status.md`, `metrics.json`) is invisible to the orchestrator/eval-agent
  afterward unless copied out. `build_sandbox_subagent()`'s `upload_paths`/`download_paths`
  do exactly that via the sandbox's own `upload_files()`/`download_files()`
  (`backends/sandboxes.py`), bracketing one inner-agent invocation — same `/workspace/...`
  absolute-path convention on both sides, so a path string means the same thing whether it
  resolves to real disk or the sandbox's container filesystem. eval-agent has no
  `upload_paths`: it's passed the SAME `training_sandbox_backend` singleton instance as
  training-agent (see `agent.py`), so the dataset/weights training-agent already staged
  there are still present — re-uploading would be redundant, matching the pre-existing
  "eval-agent reuses the training sandbox" design intent.
  One more gotcha this surfaced: a `CompiledSubAgent`'s inner `create_deep_agent(...)` is
  its own independent call, NOT a declarative `SubAgent` inside the orchestrator's shared
  list — so it does NOT automatically inherit `ORCHESTRATOR_MODEL` the way deepagents' own
  `spec.get("model", model)` mechanism does for every other subagent. Both
  `subagents/training.py` and `subagents/eval.py` now resolve their model explicitly
  (`TRAINING_AGENT_MODEL`/`EVAL_AGENT_MODEL`, falling back to `ORCHESTRATOR_MODEL`) before
  building their inner agent — omitting this silently fell back to deepagents' own
  built-in default (Anthropic) and threw a deprecation warning (`Passing model=None to
  create_deep_agent is deprecated`) until fixed.
  A live orchestrator run reaching training-agent surfaced one more real bug:
  `blockbuster.blockbuster.BlockingError: Blocking call to os.getcwd` — `_run`'s upload/
  download bridge in `sandbox_subagent.py` called blocking synchronous code (real-disk
  file I/O via `workspace_path()`/`Path.resolve()`, plus `modal`'s synchronous SDK calls)
  directly inside an async function; `langgraph dev`'s blockbuster middleware correctly
  flags any bare blocking call made straight in the event loop. Fixed by wrapping both the
  upload and download steps in `asyncio.to_thread(...)` (blockbuster's own suggested fix).
  Verified with a fake sandbox backend (no real Modal cost) that the bridge still collects/
  uploads/downloads the right paths and content under this threaded wrapping.
  **Not yet exercised against a real Modal GPU run** — spinning up a real sandbox costs
  real money/time, deliberately not triggered without an explicit go-ahead. This backend is
  written to be provider-agnostic: `deepagents` also ships a `LangSmithSandbox`
  (`deepagents/backends/langsmith.py`, wraps `langsmith.sandbox.Sandbox`) implementing the
  same `SandboxBackendProtocol` — if that's available/preferred later, swap the
  `sandbox_backend` instance passed to `build_training_agent`/`build_eval_agent`; nothing
  in `sandbox_subagent.py` is Modal-specific.
  `dataset-agent` never needed any of this: its merge/dedupe/split work is plain Python
  (`tools/dataset_builder.py`) that runs directly in the orchestrator process, no sandbox
  needed, since it's CPU-bound file/image work, not GPU training.
- `tools/zero_shot_annotate.py` — the actual zero-shot detector call (YOLO-World /
  Grounding DINO) is unimplemented and raises `NotImplementedError`. Moot while
  `annotation-agent` is excluded from the active roster; if re-enabled, it would need the
  same `build_sandbox_subagent()` treatment as training-agent/eval-agent, not the inert
  `"backend"` dict key it still uses.
- Roboflow/Kaggle MCP server URLs/credentials, Modal auth, and the Ollama Cloud key
  (`OLLAMA_API_KEY`/`OLLAMA_BASE_URL`, shared by every subagent via `tools/model_builder.py`
  — see Model wiring above) are expected to be filled into `.env` per-deployment. Note the
  Roboflow MCP URL is `https://mcp.roboflow.com/mcp`
  (needs the `/mcp` path) — the bare root URL returns `405 Method Not Allowed`; this was
  wrong in `.env.example`/`tools/mcp_clients.py`'s default until it was verified against
  a real key and fixed. Kaggle MCP's real, working endpoint is `https://www.kaggle.com/mcp`
  (confirmed: returns ~70 real tools spanning datasets/competitions/notebooks/forums) —
  the earlier "no known public endpoint" note was wrong, corrected once a real
  `KAGGLE_MCP_URL` was actually tried. `KAGGLE_API_KEY` (optional, sent as a Bearer header
  like Roboflow's) is wired in `mcp_clients.py` but unverified — tool *listing* works
  without it; whether tool *calls* need it is untested.
- `training-agent`'s `ultralytics` invocation is still only system-prompt instructions
  relying on the model to run the right `yolo detect train ...` command via `execute()` -
  no code parses/validates its output the way `tools/dataset_builder.py` does for
  dataset-agent. It now genuinely HAS a working `execute()` tool (see the backend-override
  fix above), so this is no longer blocked at the wiring level - what's untested is an
  actual live GPU run (real Modal cost/time - not triggered without explicit go-ahead) and
  whether the model reliably produces `runs/train/metrics.json` in a shape eval-agent can
  parse.
- `dataset-agent`'s merge/dedupe/split step (this is the one actually implemented — see
  the Subagent pipeline section above) still assumes each source's raw files land under
  `/workspace/sourced/<index>/` in a specific shape (`images/`, `labels/`, `classes.txt`);
  dataset-agent's system prompt instructs it to stage sources there itself via MCP
  tools/`download_and_extract`. **Now exercised against real Roboflow Universe sources
  via the full orchestrator + UI flow**, which surfaced a real gap since fixed: fetching a
  Roboflow Universe source is a 5-step async chain (`projects_fork` → poll
  `async_tasks_get` → `versions_generate` → poll `versions_get` → `versions_export` →
  `download_and_extract` on the resulting URL) — `subagents/dataset.py`'s
  `_ROBOFLOW_TOOL_ALLOWLIST` was missing `async_tasks_get` entirely (so `projects_fork`'s
  async task, which only returns a `taskId`, could never be confirmed complete), and the
  system prompt only said "use the Roboflow/Kaggle MCP tools" without spelling out the
  sequence. Both fixed: `async_tasks_get` added to the allowlist, and the prompt now
  states the exact 5-step chain (including accepting `versions_generate`'s default
  preprocessing/augmentation, since dataset-agent has no way to pause and confirm those
  choices with the user mid-run — only the orchestrator's three gates do that). Not yet
  re-verified against a real fork/export end-to-end (Roboflow's own async processing can
  take real wall-clock time) — if a run still reports a stage as blocked after this fix,
  check *which* step failed (the summary should now say) rather than assuming the same
  root cause.
