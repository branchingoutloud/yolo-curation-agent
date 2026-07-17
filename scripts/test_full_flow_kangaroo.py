"""Exercise the FULL orchestrator (planning -> sourcing -> [gate 1] ->
dataset) against a real natural-language use case, instead of any single
subagent in isolation. Unlike scripts/test_dataset_agent_real.py (which hands
dataset-agent a hand-seeded sources.json), this script starts from nothing but
a user prompt and lets planning-agent/sourcing-agent do real work: real
Roboflow Universe/Kaggle/web searches for "kangaroo" datasets, a real
sources.json, and (if a Roboflow Universe project is found) a real
projects_fork on the configured Roboflow account.

This hits gate 1 (request_approval, the sourcing/image-budget plan) as a real
LangGraph interrupt and auto-approves it so the run can proceed into
dataset-agent - it does NOT auto-approve gate 2 (model size, before training);
the script deliberately stops there since the goal is only to verify the
pipeline through a built dataset/, not to kick off a real GPU training run.

Requires a checkpointer (InMemorySaver) that agent.py's own `agent` object
does not have - langgraph dev supplies persistence via the platform instead.
This script builds its own standalone CompiledStateGraph, mirroring agent.py's
build_agent() exactly but with an explicit checkpointer, so interrupt/resume
works in a single Python process.

Usage:
    python scripts/test_full_flow_kangaroo.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Cloud models routinely emit Unicode punctuation (em-dashes, non-breaking
# hyphens) that Windows' default cp1252 console can't encode - reconfigure
# stdout to UTF-8 so printing a real model response doesn't crash the script.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_TEST_WORKSPACE = Path(__file__).resolve().parent.parent / "run_artifacts_kangaroo"
os.environ["RUN_ARTIFACTS_DIR"] = str(_TEST_WORKSPACE)

if _TEST_WORKSPACE.exists():
    import shutil

    shutil.rmtree(_TEST_WORKSPACE)
_TEST_WORKSPACE.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_TEST_WORKSPACE / "agent_run.log", encoding="utf-8", mode="w"),
    ],
)
# mcp.client.streamable_http at DEBUG dumps full tool schemas (100+ Roboflow
# tools, full multi-paragraph descriptions) as single multi-KB log lines on
# every MCP round trip - unreadable noise for a full-flow run that makes many
# of them. INFO/WARNING here; our own app loggers (below) stay informative.
for _noisy in (
    "langsmith",
    "urllib3",
    "PIL",
    "anthropic",
    "httpx",
    "httpcore",
    "asyncio",
    "deepagents",
    "mcp",
    "mcp.client.streamable_http",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from deepagents import create_deep_agent  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from backends.project_backend import project_backend  # noqa: E402
from subagents.dataset import build_dataset_agent  # noqa: E402
from subagents.eval import build_eval_agent  # noqa: E402
from subagents.planning import build_planning_agent  # noqa: E402
from subagents.sourcing import build_sourcing_agent  # noqa: E402
from subagents.training import build_training_agent  # noqa: E402
from tools.approval import request_approval  # noqa: E402
from tools.mcp_clients import get_kaggle_tools, get_roboflow_tools, get_web_search_tools  # noqa: E402
from tools.model_builder import build_model  # noqa: E402

# Mirrors agent.py's ORCHESTRATOR_PROMPT/SKILLS_DIR verbatim rather than
# importing them - importing agent.py would also execute its module-level
# `agent = build_agent()`, redundantly rebuilding a second (checkpointer-less)
# graph and re-fetching every MCP tool list a second time.
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

SKILLS_DIR = str(Path(__file__).resolve().parent.parent / "skills")

ORCHESTRATOR_MODEL_SPEC = os.environ.get("ORCHESTRATOR_MODEL", "anthropic:claude-sonnet-5")

USER_PROMPT = "I want to train a YOLO model for identifying a kangaroo."

MAX_GATES = 5


async def main(roboflow_tools: list, kaggle_tools: list, web_search_tools: list) -> None:
    logger = logging.getLogger("full_flow_test")
    logger.info("=== Full-flow test run started ===")
    logger.info("Roboflow tools: %d, Kaggle tools: %d, web_search tools: %d", len(roboflow_tools), len(kaggle_tools), len(web_search_tools))

    subagents = [
        build_planning_agent(),
        build_sourcing_agent(roboflow_tools, kaggle_tools, web_search_tools),
        build_dataset_agent(roboflow_tools, kaggle_tools),
        build_training_agent(),
        build_eval_agent(),
    ]

    orchestrator_model = build_model(ORCHESTRATOR_MODEL_SPEC) or ORCHESTRATOR_MODEL_SPEC
    logger.info("Using orchestrator model: %s", ORCHESTRATOR_MODEL_SPEC)

    full_agent = create_deep_agent(
        model=orchestrator_model,
        system_prompt=ORCHESTRATOR_PROMPT,
        subagents=subagents,
        tools=[request_approval],
        interrupt_on={
            "request_approval": {"allowed_decisions": ["approve", "edit", "reject"]},
        },
        skills=[SKILLS_DIR],
        backend=project_backend,
        checkpointer=InMemorySaver(),
    )

    config = {"configurable": {"thread_id": "kangaroo-flow-1"}, "recursion_limit": 250}
    inputs = {"messages": [HumanMessage(content=USER_PROMPT)]}

    print(f"\n>>> Sending prompt: {USER_PROMPT!r}\n")
    result = await full_agent.ainvoke(inputs, config=config)

    gate_num = 0
    while "__interrupt__" in result and gate_num < MAX_GATES:
        gate_num += 1
        interrupt_obj = result["__interrupt__"][0]
        payload = interrupt_obj.value
        action_requests = payload.get("action_requests", [])

        print(f"\n=== APPROVAL GATE {gate_num} ===")
        for ar in action_requests:
            print(f"  tool: {ar.get('name')}")
            print(f"  args: {ar.get('args')}")
            print(f"  description: {ar.get('description', '')}")

        if gate_num == 1:
            print(">>> auto-approving gate 1 (sourcing/data plan) to proceed into dataset-agent\n")
            decisions = [{"type": "approve"} for _ in action_requests]
            result = await full_agent.ainvoke(Command(resume={"decisions": decisions}), config=config)
        else:
            print(f">>> stopping here at gate {gate_num} - test scope is 'through dataset build only', not training\n")
            break

    print("\n=== FINAL MESSAGE ===")
    if "__interrupt__" in result:
        print("(graph is paused on an interrupt - see gate details above; no final AIMessage yet)")
    else:
        print(result["messages"][-1].content)

    print("\n=== sources.json ===")
    sources_path = _TEST_WORKSPACE / "sources.json"
    if sources_path.exists():
        print(sources_path.read_text(encoding="utf-8"))
    else:
        print("(no sources.json was created)")

    print("\n=== dataset/ file tree (first 60 entries) ===")
    dataset_dir = _TEST_WORKSPACE / "dataset"
    if dataset_dir.exists():
        entries = sorted(dataset_dir.rglob("*"))
        for p in entries[:60]:
            print(p.relative_to(_TEST_WORKSPACE))
        if len(entries) > 60:
            print(f"... and {len(entries) - 60} more")
    else:
        print("(no dataset/ directory was created)")

    print("\n=== sourced/ file tree (first 60 entries) ===")
    sourced_dir = _TEST_WORKSPACE / "sourced"
    if sourced_dir.exists():
        entries = sorted(sourced_dir.rglob("*"))
        for p in entries[:60]:
            print(p.relative_to(_TEST_WORKSPACE))
        if len(entries) > 60:
            print(f"... and {len(entries) - 60} more")
    else:
        print("(no sourced/ directory was created)")


if __name__ == "__main__":
    # get_roboflow_tools()/get_kaggle_tools()/get_web_search_tools() internally
    # call asyncio.run() - must happen here, before any event loop is running,
    # or it raises "cannot be called from a running event loop" (silently
    # caught by mcp_clients.py's broad except, so it just looks like zero
    # tools loaded rather than an obvious error).
    _roboflow_tools = get_roboflow_tools()
    _kaggle_tools = get_kaggle_tools()
    _web_search_tools = get_web_search_tools()
    asyncio.run(main(_roboflow_tools, _kaggle_tools, _web_search_tools))
