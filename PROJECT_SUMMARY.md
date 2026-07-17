# YOLO Curation Agent — Project Summary

Status snapshot as of 2026-07-17. Cross-referenced against `agent.py`, `subagents/`,
`tools/`, `backends/`, `CLAUDE.md`, and the full design doc,
`yolo-deep-agent-architecture (1).md` (hereafter "the architecture doc").

---

## 1. What this project is

A `deepagents`-native orchestrator agent (LangGraph-served) that turns a user's
object-detection use case into a trained, evaluated YOLO model, autonomously:
it plans a dataset budget, sources public data, assembles a YOLO-format dataset,
trains a model, evaluates it, and decides whether another iteration is warranted —
pausing for human approval at exactly three checkpoints. It is meant to showcase,
as first-class `deepagents` features rather than bolted-on code: subagents, an
agent-driven (not hardcoded) sequencing loop, a virtual filesystem for context
management, sandboxed code execution, MCP tools, Agent Skills, and
human-in-the-loop approval.

The companion UI (`../deep-agents-ui`, a separate Next.js repo) talks to this
agent's LangGraph server independently and is out of scope for this summary.

---

## 2. What is achieved (working today)

### Orchestrator (`agent.py`)
Implemented essentially as designed in the architecture doc (§3). It is a single
`create_deep_agent(...)` call with:
- **No data/training tools of its own** — its only bound tool is
  `request_approval` (`tools/approval.py`), gated via `interrupt_on` so LangGraph
  pauses/resumes natively rather than via custom prompting.
- A prompt that gives it a **goal and judgment calls**, not a fixed script —
  it uses `write_todos` to plan, delegates via `task`, reads subagent output
  files, and decides for itself whether another sourcing/annotation/training
  round is warranted.
- The three approval gates are enforced purely by prompt instruction (not code):
  (1) after a sourcing/image-budget plan, before data is pulled; (2) after a
  proposed YOLO size, before training; (3) after eval results, before proposing
  another iteration round.
- Model wiring goes through `tools/model_builder.py`'s `build_model()`, which is
  what lets an `"ollama:..."` spec pick up Ollama Cloud's base_url/Bearer-auth
  wiring, instead of a raw string handed to `init_chat_model`. The whole system
  is currently running on **Ollama Cloud** (`gpt-oss:120b` for orchestrator/
  sourcing/dataset, `gpt-oss:20b` for training/eval), with per-subagent
  `*_AGENT_MODEL` overrides all optional and falling back to `ORCHESTRATOR_MODEL`.
  A process-wide lock (`_ThrottledCloudChatOllama`) serializes calls because the
  current key is free-tier, single-concurrency.

### Active subagent roster: planning → sourcing → dataset → training → eval
`annotation-agent` exists in code (`subagents/annotation.py`) but is **excluded**
from the active roster in `agent.py` — sourcing-agent hands off directly to
dataset-agent, and dataset-agent treats any `sources.json` entry with
`status != "available"` as pending rather than assuming annotation will handle it.

| Subagent | Real logic implemented? | Verified how |
|---|---|---|
| **planning-agent** | Yes — real Python `write_plan` tool (`tools/planning_builder.py`) checks each class's count against the `cv-dataset-curation` skill's floor, flags "thin" classes, writes `plan.md` + `class_budget.json` | Live end-to-end model run (Groq, then Ollama Cloud `gpt-oss:120b`) |
| **sourcing-agent** | Yes — real `append_sources` tool (`tools/sourcing_builder.py`) reads existing `sources.json` and updates-or-appends by `(source, dataset_id)`, rejects malformed entries in code (not just by prompt) | Direct tool invocation (no LLM) + live Roboflow Universe search on Ollama Cloud — something that never completed on Groq/Gemini free tiers |
| **dataset-agent** | Yes — real `merge_and_split_dataset` tool (`tools/dataset_builder.py`): remaps class IDs per-source through each source's own `classes.txt`, dedupes via perceptual hashing (`imagehash.phash`), splits train/val/test, writes `dataset/data.yaml` | Direct tool invocation on seeded synthetic images + live end-to-end run on Ollama Cloud (correctly merged 4/5 images, dropped the true near-duplicate) |
| **training-agent** | **No** — only system-prompt instructions; the `execute()` sandbox it depends on isn't actually reachable (see §3) | Not functional |
| **eval-agent** | **No** — same sandbox gap as training-agent | Not functional |

### Skills (`skills/`)
Three skills exist and match the architecture doc's design (on-demand domain
knowledge via `SkillsMiddleware`, not hardcoded in prompts):
`yolo-model-selection`, `cv-dataset-curation`, `cv-eval-and-iteration`.

### Backends (`backends/`)
- `project_backend.py` — the shared virtual filesystem (`CompositeBackend` of
  `StateBackend` + `FilesystemBackend` rooted at `RUN_ARTIFACTS_DIR`) is working
  as designed: `plan.md`, `sources.json`, `dataset/`, etc. are real, inspectable
  files on disk. This is the working half of the "context management" story —
  subagents coordinate via files, not shared state or return values.
- `sandboxes.py` — `ModalSandbox` (wrapping the `modal` SDK) is fully implemented
  as a `BaseSandbox` subclass with lazy single-container reuse and `atexit`
  teardown. **But it is not wired to anything** (see §3).

### Testing (`scripts/`)
No pytest suite exists by design (documented in `CLAUDE.md`); instead there are
standalone harnesses per subagent (`test_planning_agent.py`,
`test_dataset_agent.py` + `seed_dummy_workspace.py`, `test_sourcing_agent.py`,
plus real-credential variants and `check_ollama.py` as an Ollama preflight
check). Planning, sourcing, and dataset-agent are all confirmed working
end-to-end against real models; several real bugs (tool-exception crashing the
graph, path-argument hallucination, staged-file detection) were found and fixed
this way.

---

## 3. How it is implemented (architecture as actually wired)

This follows the architecture doc's design closely, with one major deviation
forced by the installed library version:

- **Orchestrator + subagents** via `deepagents.create_deep_agent` /
  `SubAgent`, exactly per §3–4 of the architecture doc.
- **MCP tools**: Roboflow (`https://mcp.roboflow.com/mcp` — note the required
  `/mcp` path, wrong in early `.env.example`) and Kaggle
  (`https://www.kaggle.com/mcp`, ~70 tools) are both real, reachable MCP
  servers, trimmed to an allowlist per subagent (`_ROBOFLOW_TOOL_ALLOWLIST` /
  `_KAGGLE_TOOL_ALLOWLIST`) to avoid blowing past token limits. All optional at
  import time — a missing key just shrinks that subagent's tool list to `[]`,
  never raises (`tools/mcp_clients.py`).
- **Filesystem-mediated handoff**: every subagent writes its output to a file
  under `/workspace/...` and returns only a short summary — the same discipline
  the architecture doc specifies for context isolation.
- **Sandbox architecture mismatch (the one real gap vs. the design doc):**
  The architecture doc's design (§4, §8) calls for **per-subagent** sandbox
  backends — `annotation-agent`/`training-agent`/`eval-agent` each declare a
  `"backend"` key in their `SubAgent` dict pointing at a specific `ModalSandbox`.
  The **installed `deepagents` version (0.6.12) does not support this** —
  `SubAgent` has no `backend` field, and `create_deep_agent`'s subagent-building
  loop always binds every subagent's `FilesystemMiddleware` to the single
  top-level `backend=` passed to `create_deep_agent` (which is `project_backend`,
  not a sandbox). Consequence: **no subagent has an `execute` tool at all**,
  because `FilesystemMiddleware` only adds `execute` when the bound backend
  supports it, and `project_backend` doesn't. The `"backend"` keys in
  `annotation.py`/`training.py`/`eval.py` are silently ignored Python dict
  entries — not a config typo, a structural version gap.
  - `dataset-agent` **sidesteps this entirely** — its merge/dedupe/split work
    is plain CPU-bound Python (`tools/dataset_builder.py`) that runs directly in
    the orchestrator process, no sandbox needed. This is why dataset-agent is
    fully functional while training/eval are not.

---

## 4. What needs to be done

In rough priority order (blocking → high-value → polish):

### Blocking: the sandbox/backend gap
Nothing GPU-dependent (training, real annotation) can run until this is fixed.
Per `CLAUDE.md`'s Known-stubs section, the two realistic fixes are:
1. **Give `deepagents` a real per-subagent backend mechanism** (requires either
   upgrading past 0.6.12 if a newer release adds this, or patching/wrapping
   `SubAgentMiddleware`'s subagent-building loop to read a `"backend"` key off
   each `SubAgent` dict and bind it instead of always using the parent's).
   Check the installed version's changelog / a newer `deepagents` release first
   — this may already be solved upstream.
2. **Fallback: pass a sandbox as the single top-level `backend=`.** Simpler, but
   it would give *every* subagent (including planning/sourcing, which don't
   need it) an `execute` tool and a sandbox filesystem — a real behavior change,
   not just a wiring fix, and would need re-verifying that
   plan.md/sources.json/etc. still land in the right place.

Until one of these lands, `training-agent` and `eval-agent` are **prompt-only
stubs** — they will attempt to call `execute()` and fail, because the tool
doesn't exist on their subagent instance at all.

### High-value, unblocked by the sandbox issue
- **`tools/zero_shot_annotate.py`** — currently `raise NotImplementedError(...)`.
  Needs a real zero-shot open-vocab detector wired in (YOLO-World is suggested
  as the lighter option vs. Grounding DINO if sandbox GPU is constrained per
  §13 of the architecture doc). Moot until `annotation-agent` is both fixed
  (sandbox) and re-added to the active roster in `agent.py` — currently
  commented out.
- **Re-enable `annotation-agent`** in `agent.py`'s `subagents=[...]` list once
  (a) the sandbox issue is fixed and (b) `zero_shot_annotate` has a real model.
  The import/wiring comment in `agent.py` already documents exactly what to
  restore.
- **Verify `dataset-agent`'s live MCP staging path** — the merge/dedupe/split
  logic is solid and independently verified, but the *staging* step (fetching
  raw files into `/workspace/sourced/<i>/` via real Roboflow/Kaggle MCP tools
  or `download_and_extract`) has only been exercised with pre-seeded synthetic
  data and a real-but-network-only sourcing pass — not a full real
  source-to-download-to-merge run against live Roboflow/Kaggle credentials.
  `scripts/test_dataset_agent_real.py` / `seed_real_sources.py` exist for this;
  actually running them against real credentials is the open item.
- **`KAGGLE_API_KEY` behavior is unverified** — tool *listing* works without it;
  whether tool *calls* (actual downloads) require it is untested.

### Medium-value / robustness
- Once training-agent is unblocked, do a real cached end-to-end training run
  (per the architecture doc §12/§13 risk mitigation: pre-run once, cache a
  checkpoint, so a live demo only needs a short fine-tune / few epochs rather
  than a full run under time pressure).
- Wire LangSmith tracing to make the context-management demo point visible
  (orchestrator trace flat/short vs. training-agent's trace long) — mentioned
  as a demo beat in the architecture doc (§7) but not confirmed configured in
  this repo.
- HuggingFace fallback model provider is documented but genuinely unwired (no
  `langchain-huggingface` dependency, no token) — only worth doing if the
  Ollama Cloud single-concurrency free-tier key becomes an actual bottleneck.

### Low-value / explicitly out of scope for now
- No lint/test/CI tooling is configured, and `CLAUDE.md` treats "does
  `langgraph dev` boot and does the graph run" as the current bar — introducing
  a formal test suite is a reasonable ask but not blocking the core demo.

---

## 5. How it should be done (recommended approach)

1. **Resolve the sandbox/backend gap first** — it blocks training-agent,
   eval-agent, and annotation-agent simultaneously, i.e. the entire second half
   of the pipeline. Check whether a newer `deepagents` release already supports
   per-subagent backends before hand-rolling a patch; that's strictly less risky
   than modifying `SubAgentMiddleware` behavior directly. If patching is
   unavoidable, keep the patch isolated (e.g. a thin wrapper around
   `create_deep_agent` in `agent.py`) so it's easy to delete once upstream
   support lands.
2. **Get one real, cached, end-to-end training run working** before doing
   anything else with training-agent — per the architecture doc's own risk
   list, live GPU spin-up latency during a demo is a known risk, and a cached
   checkpoint is the documented mitigation. Do this even at reduced epoch count.
3. **Only then wire zero-shot annotation and re-enable annotation-agent** —
   it's the least-load-bearing piece of the demo narrative (sourcing-agent
   already covers "does the pipeline find real data") and depends on the same
   sandbox fix, so sequencing it last avoids re-doing work.
4. **Keep using the existing no-LLM-first testing discipline** — every custom
   `@tool` (like `merge_and_split_dataset`/`append_sources`) should be verified
   by direct `.invoke()`/`.ainvoke()` against seeded files before ever spending
   a model call on it. This has already caught real bugs (uncaught exceptions
   crashing the whole graph run) cheaply; keep doing it for any new tool,
   especially inside the sandbox-execute path once that's unblocked.
5. **Don't restore per-subagent `"backend"` dict keys as decoration** — if the
   sandbox fix ends up being "single top-level backend for everyone," remove
   the now-misleading `"backend": sandbox_backend` keys from
   `annotation.py`/`training.py`/`eval.py` rather than leaving inert fields
   that look like they do something.
6. **Preserve the three-gate contract** while making any of the above changes —
   it's enforced only by the orchestrator's system prompt, not by code, so it's
   easy to accidentally erode when editing `ORCHESTRATOR_PROMPT` for unrelated
   reasons (e.g. adding annotation-agent back to the roster).
