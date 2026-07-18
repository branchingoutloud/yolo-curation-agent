"""Exercise dataset-agent against REAL Roboflow Universe sources (see
scripts/seed_real_sources.py) instead of synthetic dummy data.

Two very different things can happen depending on your .env:

  - ROBOFLOW_API_KEY unset: dataset-agent gets zero Roboflow tools and no
    locally-staged files, so it correctly reports both sources as "not staged
    yet, nothing to merge" - this is the EXPECTED, correct behavior, not a
    bug, and is a legitimate test of the skip/report path on realistic data.

  - ROBOFLOW_API_KEY set: dataset-agent has real Roboflow MCP tools and may
    attempt to actually fetch and merge ~2,850 real images across both
    datasets. This can take a while and use real bandwidth/disk - there is no
    hard cap in this script, so watch the output.

Usage:
    python scripts/test_dataset_agent_real.py
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

_TEST_WORKSPACE = Path(__file__).resolve().parent.parent / "run_artifacts_real_test"
os.environ["RUN_ARTIFACTS_DIR"] = str(_TEST_WORKSPACE)

_TEST_WORKSPACE.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_TEST_WORKSPACE / "agent_run.log", encoding="utf-8", mode="w"),
    ],
)
for _noisy in ("langsmith", "urllib3", "PIL", "anthropic", "httpx", "httpcore", "asyncio", "deepagents"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from deepagents import create_deep_agent  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

from backends.project_backend import project_backend  # noqa: E402
from scripts.seed_real_sources import seed_real_sources  # noqa: E402
from subagents.dataset import build_dataset_agent  # noqa: E402
from tools.mcp_clients import get_kaggle_tools, get_roboflow_tools  # noqa: E402
from tools.model_builder import build_model  # noqa: E402

async def main(roboflow_tools: list, kaggle_tools: list):
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            root_logger.removeHandler(handler)

    if _TEST_WORKSPACE.exists():
        import shutil
        shutil.rmtree(_TEST_WORKSPACE)
    seed_real_sources(_TEST_WORKSPACE)

    file_handler = logging.FileHandler(_TEST_WORKSPACE / "agent_run.log", encoding="utf-8", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    root_logger.addHandler(file_handler)
    logger = logging.getLogger("dataset_agent")
    logger.info("=== Test run started ===")

    print(f"\nRoboflow tools available: {len(roboflow_tools)} {'(set ROBOFLOW_API_KEY in .env to enable)' if not roboflow_tools else ''}")
    print(f"Kaggle tools available: {len(kaggle_tools)}\n")

    model_spec = os.environ.get("DATASET_AGENT_MODEL") or os.environ.get("ORCHESTRATOR_MODEL", "anthropic:claude-sonnet-4-6")
    model = build_model(model_spec) or model_spec
    logger.info("Using model: %s (resolved from %s)", model, model_spec)

    spec = build_dataset_agent(roboflow_tools=roboflow_tools, kaggle_tools=kaggle_tools)
    print(f"Model: {model_spec}\n")

    test_agent = create_deep_agent(
        model=model,
        system_prompt=spec["system_prompt"],
        tools=spec["tools"],
        backend=project_backend,
    )

    # Real Roboflow/Kaggle MCP tools are async-only (StructuredTool with only
    # _arun implemented) - sync .invoke() raises "does not support sync
    # invocation" the moment the LLM calls one. langgraph dev serves this
    # graph asynchronously already, so this is only a test-script
    # requirement, not a production behavior change.
    result = await test_agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Read sources.json and merge whatever is available into a "
                        "training dataset. If a source isn't staged locally yet, "
                        "try to fetch it with whatever tools you have; if you can't, "
                        "say so plainly rather than guessing. Report what you did."
                    )
                )
            ]
        },
        config={"recursion_limit": 300},
    )

    print("=== FINAL MESSAGE ===")
    print(result["messages"][-1].content)

    print("\n=== dataset/ file tree (first 40 entries) ===")
    dataset_dir = _TEST_WORKSPACE / "dataset"
    if dataset_dir.exists():
        entries = sorted(dataset_dir.rglob("*"))
        for p in entries[:40]:
            print(p.relative_to(_TEST_WORKSPACE))
        if len(entries) > 40:
            print(f"... and {len(entries) - 40} more")
    else:
        print("(no dataset/ directory was created)")

    print("\n=== sourced/ file tree (raw staged files, if any) ===")
    sourced_dir = _TEST_WORKSPACE / "sourced"
    if sourced_dir.exists():
        entries = sorted(sourced_dir.rglob("*"))
        for p in entries[:40]:
            print(p.relative_to(_TEST_WORKSPACE))
        if len(entries) > 40:
            print(f"... and {len(entries) - 40} more")
    else:
        print("(no sourced/ directory was created - nothing was fetched)")


if __name__ == "__main__":
    # get_roboflow_tools()/get_kaggle_tools() internally call asyncio.run() -
    # must happen here, before any event loop is running, or it raises
    # "asyncio.run() cannot be called from a running event loop" and silently
    # degrades to zero tools (caught by mcp_clients.py's broad except).
    # agent.py doesn't hit this: it builds tools at plain synchronous import
    # time, before langgraph dev's server loop starts - this ordering
    # requirement is specific to this script's async structure.
    _roboflow_tools = get_roboflow_tools()
    _kaggle_tools = get_kaggle_tools()
    asyncio.run(main(_roboflow_tools, _kaggle_tools))
