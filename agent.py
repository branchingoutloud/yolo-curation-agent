"""Orchestrator wiring for the YOLO curation & training deep agent.

The exported `agent` is what langgraph.json points at (./agent.py:agent).
See yolo-deep-agent-architecture.md for the full design rationale - this
file is the §3 code sample wired up to the actual subagents/tools/backends
modules in this repo instead of inline placeholders.
"""

import os
from pathlib import Path

from deepagents import create_deep_agent

from backends.project_backend import project_backend
from subagents.dataset import build_dataset_agent
from subagents.eval import build_eval_agent
from subagents.planning import build_planning_agent
from subagents.sourcing import build_sourcing_agent
from subagents.training import build_training_agent
from tools.approval import request_approval
from tools.mcp_clients import get_kaggle_tools, get_roboflow_tools, get_web_search_tools
from tools.model_builder import build_model

# annotation-agent is temporarily excluded from the active roster - sourcing-agent
# now hands off directly to dataset-agent. dataset-agent treats any sources.json
# entry with status != "available" as pending (not routed anywhere) rather than
# assuming annotation-agent will pick it up. To re-enable: restore the
# `from subagents.annotation import build_annotation_agent`,
# `from backends.sandboxes import annotation_sandbox_backend`, and
# `from tools.zero_shot_annotate import zero_shot_annotate` imports and add
# `build_annotation_agent(roboflow_tools, [zero_shot_annotate], annotation_sandbox_backend)`
# back into the `subagents` list below - note its "backend" override won't
# actually take effect either way (see tools/dataset_builder.py's module
# docstring for why per-subagent backend overrides are inert in
# deepagents==0.6.12).

ORCHESTRATOR_PROMPT = """
You are the orchestrator for a YOLO data-curation and training agent. Your goal:
take a user's detection use case and produce a trained, evaluated YOLO model,
improving it over as many iterations as the data and eval results warrant.

You have no data-sourcing, annotation, or training abilities yourself - you only
plan, delegate to subagents via `task`, read their output files, reason about
whether the result is good enough, and either delegate further or ask the user
to weigh in via `request_approval`.

Use `write_todos` to track your own plan and update it as you learn more -
do not assume the pipeline only runs once. After every eval-agent result,
decide for yourself whether another sourcing/annotation/training round is
warranted, and say why, before proposing it to the user.

Call `request_approval` (and only then) at these three natural checkpoints:
(1) once you have a sourcing/image-budget plan you're confident in and before
    any data is actually pulled or forked into a workspace,
(2) once you've proposed a specific YOLO model size and before training starts,
(3) once you have eval results and a concrete, reasoned iteration plan.
Do not call `request_approval` at any other time, and do not skip these three.

Never fabricate metrics, dataset stats, or file contents - always read them
from the filesystem tools first.
"""

SKILLS_DIR = str(Path(__file__).resolve().parent / "skills")

ORCHESTRATOR_MODEL_SPEC = os.environ.get("ORCHESTRATOR_MODEL", "anthropic:claude-sonnet-5")


def build_agent():
    roboflow_tools = get_roboflow_tools()
    kaggle_tools = get_kaggle_tools()
    web_search_tools = get_web_search_tools()

    subagents = [
        build_planning_agent(),
        build_sourcing_agent(roboflow_tools, kaggle_tools, web_search_tools),
        build_dataset_agent(roboflow_tools, kaggle_tools),
        build_training_agent(),
        build_eval_agent(),
    ]

    # build_model resolves "ollama:..." through the shared Ollama-Cloud-aware
    # helper (base_url/API-key/single-concurrency-lock wiring - see
    # tools/model_builder.py); anything else (anthropic:, groq:, huggingface:)
    # passes straight through to init_chat_model. Falls back to the raw spec
    # string if construction fails, matching create_deep_agent's own ability
    # to accept either a resolved model object or a plain string.
    orchestrator_model = build_model(ORCHESTRATOR_MODEL_SPEC) or ORCHESTRATOR_MODEL_SPEC

    return create_deep_agent(
        model=orchestrator_model,
        system_prompt=ORCHESTRATOR_PROMPT,
        subagents=subagents,
        tools=[request_approval],
        interrupt_on={
            "request_approval": {"allowed_decisions": ["approve", "edit", "reject"]},
        },
        skills=[SKILLS_DIR],
        backend=project_backend,
    )


agent = build_agent()
