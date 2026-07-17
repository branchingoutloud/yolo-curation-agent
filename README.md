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
subagents/            # planning, sourcing, annotation, dataset, training, eval
skills/               # yolo-model-selection, cv-dataset-curation, cv-eval-and-iteration
tools/                # MCP client loading, zero_shot_annotate, request_approval
backends/             # virtual filesystem (project_backend) + sandbox configs (sandboxes)
run_artifacts/        # persisted plan.md, sources.json, dataset/, runs/, eval_report.md
langgraph.json         # graph key "yolo-agent" -> ./agent.py:agent
```

## Setup

```bash
pip install -e .
# or: uv pip install -e .
pip install -U "langgraph-cli[inmem]"

cp .env.example .env
# fill in ANTHROPIC_API_KEY at minimum; ROBOFLOW_API_KEY / KAGGLE_MCP_URL /
# TAVILY_API_KEY are optional - missing ones just degrade that subagent's
# tool list instead of failing to boot (see tools/mcp_clients.py).
```

### Running against Ollama (local dev / testing)

`ORCHESTRATOR_MODEL` accepts any `provider:model` string, but `"ollama:..."`
is special-cased in `tools/models.py`: Ollama Cloud needs a custom
`base_url` + `Authorization: Bearer <key>` header that plain
`init_chat_model("ollama:...")` can't supply, so a real `ChatOllama` object
is constructed once and reused across the orchestrator and every subagent
that doesn't set its own `model`.

In `.env`:

```bash
ORCHESTRATOR_MODEL=ollama:gpt-oss:20b   # or any model your Ollama Cloud account has
OLLAMA_API_KEY=...
OLLAMA_BASE_URL=https://ollama.com
```

Sanity check the Ollama connection alone (no deepagents involved):
`python ollama_test.py`.

### Sandbox (execute()) — do you need Docker?

`annotation-agent`/`training-agent`/`eval-agent` get a shell via `execute()`.
`SANDBOX_BACKEND` in `.env` picks how:

- **`local`** (default) — runs directly on this machine via deepagents'
  `LocalShellBackend`. No setup, uses the `ultralytics`/`torch` already in
  this venv. This is what the test scripts below use. **No isolation** —
  fine for solo local dev, not for anything else (see the security warning
  in `deepagents.backends.local_shell`).
- **`docker`** — runs inside a container (`docker/Dockerfile` +
  `docker/docker-compose.yml`) via a small custom `DockerSandbox` backend
  (`backends/docker_sandbox.py`) — deepagents itself ships no Docker/Modal
  sandbox. Real isolation; requires Docker Desktop running.

  ```bash
  docker compose -f docker/docker-compose.yml up -d --build
  # then in .env: SANDBOX_BACKEND=docker
  ```

  The Dockerfile is CPU-only (`python:3.11-slim` + `ultralytics`) since a
  typical dev laptop has no NVIDIA GPU; swap the base image and uncomment
  the `deploy.resources` block in `docker-compose.yml` if you have one.

## Testing the planning and training subagents

Both scripts build a *minimal* orchestrator wired to only one subagent, so
each one cleanly exercises `User -> Orchestrator -> task() -> Subagent`
(and, for training, `-> execute()`) without the rest of the pipeline in the
way. Requires `OLLAMA_API_KEY`/`OLLAMA_BASE_URL` in `.env`.

```bash
python test_planning_agent.py
# -> task("planning-agent") -> writes plan.md, class_budget.json

python test_training_agent.py
# -> stages a tiny synthetic dataset + model_choice.json as fixtures,
#    task("training-agent") -> runs a REAL 1-epoch `yolo` training job
#    (model=yolo11n.yaml, no pretrained-weight download needed) via
#    execute() -> writes runs/train/status.md, runs/train/metrics.json
```

Both exit `0` on pass, `1` on a failed assertion, `2` if required env vars
are missing.

## Run

```bash
langgraph dev
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

## Current state / TODOs

The orchestrator, all six subagents, the three HITL approval gates, the
shared filesystem+sandbox backend, and the three skills are wired up and
`agent.py` imports cleanly. Planning and training are covered by
`test_planning_agent.py`/`test_training_agent.py`. Remaining stubs:

- `tools/zero_shot_annotate.py` — the actual zero-shot detector call
  (YOLO-World / Grounding DINO) is unimplemented (raises `NotImplementedError`).
- Roboflow/Kaggle MCP server URLs and credentials in `.env` (sourcing/annotation
  degrade to an empty tool list without them — see `tools/mcp_clients.py`).
- `dataset-agent`'s merge/dedupe/split logic and `eval-agent`'s confusion-matrix
  work are described in their system prompts but rely on the model +
  `execute()` sandbox to carry them out, and aren't covered by a test script
  yet — untested end-to-end.

Two version-specific gaps worth knowing if you touch `backends/` or
`subagents/`:

- Installed deepagents (0.6.12) has **no per-subagent `backend` override** —
  a `"backend"` key on a `SubAgent` dict is accepted by the TypedDict but
  never read by `deepagents.middleware.subagents`/`deepagents.graph`, so it's
  a silent no-op. There is exactly one sandbox for the whole graph, set via
  `create_deep_agent(backend=...)` in `agent.py` (see
  `backends/project_backend.py` and `SANDBOX_BACKEND` above).
- `CompositeBackend.execute()` always delegates to `.default`, never to a
  routed backend — so `default` has to be the sandbox-capable backend itself,
  not an ephemeral `StateBackend`, or no subagent gets a working `execute()`
  tool at all regardless of what's registered under `/workspace/`.
