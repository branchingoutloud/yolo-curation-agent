# YOLO Deep Agent

An autonomous `deepagents`-native system that takes a user's detection use case,
sources and curates training data, trains a YOLO model, evaluates it, and proposes
an iteration. See `yolo-deep-agent-architecture.md` (design doc, not in this repo) for
the full rationale; this README covers running it.

The UI for this agent lives in a **sibling repo**, `deep-agents-ui/` — a separate
Node/Next.js project, run independently.

## Layout

```
agent.py             # create_deep_agent() wiring, orchestrator prompt, exports `agent`
subagents/            # planning, sourcing, dataset, training, eval (annotation.py exists but is not wired in)
skills/               # yolo-model-selection, cv-dataset-curation, cv-eval-and-iteration
tools/                # MCP client loading, dataset_builder, zero_shot_annotate, request_approval
backends/             # virtual filesystem (project_backend) + sandbox configs (sandboxes)
run_artifacts/        # persisted plan.md, sources.json, dataset/, runs/, eval_report.md
langgraph.json         # graph key "yolo-agent" -> ./agent.py:agent
```

## Setup

```bash
uv sync
```

This creates/updates `.venv` from `pyproject.toml` + `uv.lock` (includes
`langgraph-cli[inmem]`, since it's a listed dependency) - no separate `pip
install` step needed.

```bash
cp .env.example .env
# fill in ANTHROPIC_API_KEY at minimum; ROBOFLOW_API_KEY / KAGGLE_MCP_URL /
# TAVILY_API_KEY / MODAL_TOKEN_ID+SECRET are optional - missing ones just
# degrade that subagent's tool list instead of failing to boot (see
# tools/mcp_clients.py).
```

## Run

```bash
uv run langgraph dev
```

(`uv run` executes inside the project's `.venv` without needing to activate it
first; `langgraph dev` also works directly once the venv is activated.)

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

Subagents are developed and verified independently before orchestrator
integration. The sourcing agent is first — see `docs/sourcing-subagent.md`:

```bash
.venv/bin/python -m pytest tests/ -q                        # contract tests, no keys
python scripts/run_sourcing_isolated.py --mock --fresh      # behavioral, mocked search
python scripts/run_sourcing_isolated.py --fresh             # real Roboflow/Kaggle MCP
```

## Current state / TODOs

This scaffold wires up the orchestrator, five active subagents (planning,
sourcing, dataset, training, eval - annotation-agent exists but is currently
excluded from the roster), the three HITL approval gates, the shared
filesystem backend, and the three skills. `dataset-agent` has a real,
tested implementation (`tools/dataset_builder.py`: fetch, class-ID remap,
perceptual-hash dedup, split, `data.yaml`) - but a few other pieces are still
stubs to fill in before a real end-to-end run:

- **Sandbox wiring is currently a no-op for every subagent.** The installed
  `deepagents` version doesn't support per-subagent `"backend"` overrides, so
  `training-agent`/`eval-agent` have no `execute` tool at all right now
  despite `sandbox_backend` being threaded into their factory functions - see
  CLAUDE.md's Known Stubs section for the full explanation before assuming
  training can run.
- `tools/zero_shot_annotate.py` — the actual zero-shot detector call
  (YOLO-World / Grounding DINO) is unimplemented (raises `NotImplementedError`).
  Moot while annotation-agent is excluded from the roster.
- Roboflow/Kaggle MCP server URLs and credentials in `.env`.
- Modal auth (`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`) for the training sandbox.
- Model wiring runs on Ollama Cloud (`OLLAMA_API_KEY` + `OLLAMA_BASE_URL=
  https://ollama.com`) via `tools/model_builder.py` — every subagent's
  `*_AGENT_MODEL` override (or inherited `ORCHESTRATOR_MODEL`) goes through the
  same helper, which also throttles cloud calls to this account's single-
  concurrency free-tier limit. Verified live end-to-end against real Roboflow
  Universe data for both sourcing-agent and dataset-agent - see CLAUDE.md's
  Model wiring and Testing sections.
- `training-agent`'s `ultralytics` invocation is described in its system
  prompt but relies on the `execute()` sandbox to carry it out — blocked on
  the sandbox-wiring issue above, not just untested.
