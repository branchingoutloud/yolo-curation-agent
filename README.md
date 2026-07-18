# YOLO Deep Agent

An autonomous, `deepagents`-native orchestrator that takes a plain-language object-detection
use case ("detect forklifts and pallets in warehouse footage") and runs the whole
model-building loop for you: plan the dataset, source and curate training data, train a
YOLO model, evaluate it, and propose the next iteration — pausing for your approval at
three key checkpoints.

## Why

Training a YOLO model for a new use case is rarely hard because of the training itself —
it's hard because of everything around it, and because you have to do it **all over again
for every new use case**:

- **Data hunting is repetitive.** Every project starts with the same slog through
  Roboflow Universe, Kaggle, and the open web, judging licenses, label formats, class
  coverage, and image quality by hand.
- **Curation is fiddly and error-prone.** Merging datasets means remapping class IDs
  across conflicting `classes.txt` files, deduplicating near-identical images, and
  splitting train/val/test without leakage — the same mechanical work, every time.
- **Iteration never ends after one pass.** The first trained model is a diagnostic, not a
  deliverable: weak classes need more data, confused classes need cleaner labels, and each
  fix means re-sourcing, re-merging, and re-training the loop again.

This agent automates that loop end-to-end. A team of specialized subagents (planning,
sourcing, dataset curation, training, evaluation) does the repetitive work; you stay in
control through three human-in-the-loop approval gates — data budget, model size, and
whether the eval results justify another round.

See `yolo-deep-agent-architecture.md` (design doc, not checked into this repo) for the
full rationale; this README covers running it.

The UI lives in a **sibling repo**, `../deep-agents-ui` — a separate Node/Next.js project,
run independently and pointed at this agent's LangGraph server.

## How it works

The orchestrator (`agent.py`) has no data or training tools of its own — its only
non-delegation tool is `request_approval`. Everything else happens by delegating to
subagents, which coordinate purely through files on a shared virtual filesystem
(persisted to `run_artifacts/` on disk):

1. **planning-agent** — estimates images-per-class from class variability, writes
   `plan.md` + `class_budget.json`.
2. **sourcing-agent** — searches Roboflow Universe / Kaggle / web for existing datasets,
   catalogs findings in `sources.json` (append-only, code-enforced).
3. **dataset-agent** — downloads available YOLO-format sources, remaps class IDs to one
   canonical list, drops near-duplicates via perceptual hashing, splits train/val/test,
   writes `dataset/data.yaml` + the YOLO directory tree.
4. **training-agent** — picks a model size, runs real ultralytics training inside the
   official Docker image on the local GPU (`docker run --gpus all`), writes
   `runs/train/metrics.json` + `weights/best.pt`.
5. **eval-agent** — runs validation in the same GPU image: per-class mAP, confusion
   matrix, likely-cause heuristics per weak class; writes `eval_report.md` +
   `weak_classes.json`. The *orchestrator* decides what to do next.

Three approval gates (enforced via the `request_approval` tool):

1. after the sourcing/image-budget plan, before any data is pulled,
2. after the proposed YOLO model size, before training starts,
3. after eval results, before another sourcing/annotation/training round.

Domain heuristics (images-per-class floors, model-size decision rules, eval
interpretation) live in three skills under `skills/` rather than being hardcoded in
prompts — edit the skill's `SKILL.md` to tune them.

## Layout

```
agent.py              # create_deep_agent() wiring, orchestrator prompt, exports `agent`
subagents/            # planning, sourcing, dataset, training, eval
                      # (annotation.py exists but is not in the active roster)
skills/               # yolo-model-selection, cv-dataset-curation, cv-eval-and-iteration
tools/                # custom @tool implementations: planning_builder, sourcing_builder,
                      # dataset_builder, training_runner, eval_runner, approval,
                      # model_builder (Ollama Cloud wiring), mcp_clients, gpu_exec
tools/gpu_drivers/    # torch/ultralytics code that runs ONLY inside the GPU container
backends/             # project_backend (virtual FS -> run_artifacts/), sandboxes (legacy Modal)
scripts/              # standalone harnesses for testing subagents in isolation
tests/                # contract tests (no API keys needed)
run_artifacts/        # persisted plan.md, sources.json, dataset/, runs/, eval_report.md
langgraph.json        # graph key "yolo-agent" -> ./agent.py:agent
```

## Setup

```bash
uv sync
```

This creates/updates `.venv` from `pyproject.toml` + `uv.lock` (includes
`langgraph-cli[inmem]`) — no separate `pip install` step needed.

```bash
cp .env.example .env
```

Only `ANTHROPIC_API_KEY` (or an `OLLAMA_API_KEY` if `ORCHESTRATOR_MODEL` points at
Ollama) is required to boot. Everything else degrades gracefully: a missing
`ROBOFLOW_API_KEY` / `KAGGLE_MCP_URL` / `TAVILY_API_KEY` just shrinks that subagent's
tool list instead of failing at startup (see `tools/mcp_clients.py`).

### Model wiring

All model specs (`ORCHESTRATOR_MODEL` plus optional per-subagent `*_AGENT_MODEL`
overrides) go through `tools/model_builder.py`, which handles Ollama Cloud
base URL/auth and — on a free-tier key — serializes cloud calls behind a
process-wide lock (free tier allows only one in-flight request) and retries
transient 5xx errors. Preflight-check your model config before a long run:

```bash
uv run python scripts/check_ollama.py
```

### GPU prerequisites (training/eval only)

Training and eval shell out to the official ultralytics Docker image on the
**local** GPU — run `langgraph dev` on the GPU machine itself:

- NVIDIA GPU + Docker + NVIDIA Container Toolkit
- `docker pull ultralytics/ultralytics:latest`
- verify with `uv run python scripts/test_training_runner.py` (no LLM involved)

`ultralytics`/torch are deliberately **not** local dependencies — they only ever run
inside the container.

## Run

```bash
uv run langgraph dev
```

Expect:

```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

Sanity check without any UI:

```bash
curl http://127.0.0.1:2024/assistants/search -X POST -H "Content-Type: application/json" -d '{}'
```

should return an assistant with ID `yolo-agent`.

Then run the UI (`../deep-agents-ui`, `yarn dev`) and connect with:
- Deployment URL: `http://127.0.0.1:2024`
- Assistant ID: `yolo-agent`

## Testing subagents in isolation

There's no pytest-driven CI; verification is layered instead — pure-logic first, then
standalone per-subagent harnesses (each builds the real subagent spec via
`create_deep_agent`, minus the orchestrator layer):

```bash
.venv/bin/python -m pytest tests/ -q                     # contract tests, no keys needed
uv run python scripts/test_planning_agent.py             # planning, no network needed
uv run python scripts/seed_dummy_workspace.py            # then:
uv run python scripts/test_dataset_agent.py              # merge/dedupe/split, no network
uv run python scripts/test_sourcing_agent.py             # real Roboflow/Kaggle MCP search
uv run python scripts/test_training_runner.py            # GPU Docker runner, no LLM
uv run python scripts/run_sourcing_isolated.py --mock    # behavioral, mocked search
```

See `docs/sourcing-subagent.md` for the sourcing-agent deep dive. When a custom tool's
logic can be tested without an LLM (e.g. `merge_and_split_dataset`, `append_sources`),
that's done first by direct `.invoke()` — cheaper and more deterministic than a live
agent run.

## Current state

**Verified live end-to-end** (on Ollama Cloud `gpt-oss:120b`, against real Roboflow
Universe data): planning-agent, sourcing-agent, and dataset-agent, plus the full
orchestrator + UI flow through the first two approval gates.

**Wired but not yet verified live:** training-agent and eval-agent. The code path is
real (`run_training`/`run_eval` → `docker run --gpus all`), but needs a run on a GPU
host — see GPU prerequisites above.

**Known gaps** (full detail in `CLAUDE.md`'s Known Stubs section):

- **annotation-agent is excluded from the active roster.** Sources with
  `status != "available"` are left pending by dataset-agent rather than auto-labeled.
  Its zero-shot detector call (`tools/zero_shot_annotate.py`) is unimplemented, and it
  still assumes a per-subagent sandbox backend that the installed `deepagents` version
  silently ignores.
- **`ModalSandbox` (`backends/sandboxes.py`) is legacy.** The active training path uses
  the local-GPU Docker runner instead; Modal remains in-tree as an alternative GPU
  backend for whoever wants to wire it into `tools/gpu_exec.py`.
- Roboflow's full fork → generate → export async chain for dataset fetching is wired
  but not yet re-verified end-to-end against a real fork/export.
