"""Exercise dataset-agent end-to-end (real model + real tool calls) against
seeded dummy data, without needing real Roboflow/Kaggle credentials, Modal,
or a reachable Ollama server.

Usage:
    python scripts/test_dataset_agent.py

Uses an ISOLATED workspace (./run_artifacts_test, not ./run_artifacts) so this
never touches real run data - safe to re-run any time. Requires whatever
DATASET_AGENT_MODEL is set to in .env to actually be reachable (defaults to
Groq, which just needs GROQ_API_KEY - already configured for this project).
"""

import asyncio
import os
import sys
from pathlib import Path

# Cloud models routinely emit Unicode punctuation (em-dashes, non-breaking
# hyphens) that Windows' default cp1252 console can't encode - reconfigure
# stdout to UTF-8 so printing a real model response doesn't crash the script.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Must happen before importing anything that touches backends.project_backend
# (it reads RUN_ARTIFACTS_DIR at import time) - this isolates the test run
# from real ./run_artifacts.
_TEST_WORKSPACE = Path(__file__).resolve().parent.parent / "run_artifacts_test"
os.environ["RUN_ARTIFACTS_DIR"] = str(_TEST_WORKSPACE)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # won't override RUN_ARTIFACTS_DIR above (override=False default)

from deepagents import create_deep_agent  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

from backends.project_backend import project_backend  # noqa: E402
from scripts.seed_dummy_workspace import seed_dummy_workspace  # noqa: E402
from subagents.dataset import build_dataset_agent  # noqa: E402

async def main():
    if _TEST_WORKSPACE.exists():
        import shutil

        shutil.rmtree(_TEST_WORKSPACE)
    seed_dummy_workspace(_TEST_WORKSPACE)

    spec = build_dataset_agent(roboflow_tools=[], kaggle_tools=[])
    print(f"\nModel: {spec.get('model', 'inherits default')}\n")

    # A standalone create_deep_agent with no subagents of its own exercises
    # dataset-agent's real system_prompt/tools/model exactly as the
    # orchestrator's `task` tool would, minus the parent-agent/HITL-gate layer
    # - the same filesystem tools + custom tools + shared backend either way.
    test_agent = create_deep_agent(
        model=spec.get("model"),
        system_prompt=spec["system_prompt"],
        tools=spec["tools"],
        backend=project_backend,
    )

    # ainvoke, not invoke: real Roboflow/Kaggle MCP tools are async-only and
    # raise "StructuredTool does not support sync invocation" under .invoke().
    # This test currently passes empty tool lists so it wouldn't hit that, but
    # keeping it async-consistent with test_dataset_agent_real.py avoids the
    # same surprise the moment real tools are added here too.
    result = await test_agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Read sources.json and merge whatever is available into a "
                        "training dataset. Report what you did."
                    )
                )
            ]
        },
        config={"recursion_limit": 50},
    )

    print("=== FINAL MESSAGE ===")
    print(result["messages"][-1].content)

    print("\n=== dataset/ file tree ===")
    dataset_dir = _TEST_WORKSPACE / "dataset"
    if dataset_dir.exists():
        for p in sorted(dataset_dir.rglob("*")):
            print(p.relative_to(_TEST_WORKSPACE))
    else:
        print("(no dataset/ directory was created)")


if __name__ == "__main__":
    asyncio.run(main())
