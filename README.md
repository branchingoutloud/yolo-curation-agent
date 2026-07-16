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
# TAVILY_API_KEY / MODAL_TOKEN_ID+SECRET are optional - missing ones just
# degrade that subagent's tool list instead of failing to boot (see
# tools/mcp_clients.py).
```

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

This scaffold wires up the orchestrator, all six subagents, the three HITL
approval gates, the shared filesystem backend, and the three skills — but a
few pieces are stubs to fill in before a real end-to-end run:

- `tools/zero_shot_annotate.py` — the actual zero-shot detector call
  (YOLO-World / Grounding DINO) is unimplemented (raises `NotImplementedError`).
- Roboflow/Kaggle MCP server URLs and credentials in `.env`.
- Modal auth (`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`) for the training and
  annotation sandboxes.
- `dataset-agent`'s merge/dedupe/split logic and `training-agent`'s
  `ultralytics` invocation are described in their system prompts but rely on
  the model + `execute()` sandbox to carry them out — verify a real training
  run end-to-end before a live demo (per the architecture doc's build order).
