"""Exercise planning-agent end-to-end (real model + real write_plan tool
call) against a dummy use case. No external network dependency - write_plan
is pure local file writes.

Usage:
    python scripts/test_planning_agent.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Cloud models routinely emit Unicode punctuation (em-dashes, non-breaking
# hyphens) that Windows' default cp1252 console can't encode - reconfigure
# stdout to UTF-8 so printing a real model response doesn't crash the script.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_TEST_WORKSPACE = Path(__file__).resolve().parent.parent / "run_artifacts_test"
os.environ["RUN_ARTIFACTS_DIR"] = str(_TEST_WORKSPACE)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from deepagents import create_deep_agent  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

from backends.project_backend import project_backend  # noqa: E402
from subagents.planning import build_planning_agent  # noqa: E402
from tools.model_builder import build_model  # noqa: E402


async def main():
    if _TEST_WORKSPACE.exists():
        import shutil

        shutil.rmtree(_TEST_WORKSPACE)
    _TEST_WORKSPACE.mkdir(parents=True)

    spec = build_planning_agent()

    # planning-agent has no model override of its own - in the real
    # orchestrator it inherits ORCHESTRATOR_MODEL (see agent.py). Mirror that
    # via build_model (not a raw string) so this test gets the same Ollama
    # Cloud base_url/API-key/concurrency-lock wiring the real agent gets -
    # falls back to Groq if ORCHESTRATOR_MODEL is unset.
    orchestrator_model_spec = os.environ.get("ORCHESTRATOR_MODEL", "groq:openai/gpt-oss-120b")
    test_agent = create_deep_agent(
        model=build_model(orchestrator_model_spec) or orchestrator_model_spec,
        system_prompt=spec["system_prompt"],
        tools=spec["tools"],
        backend=project_backend,
    )

    result = await test_agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Use case: detect forklifts and pedestrians in warehouse "
                        "safety camera footage, to flag near-misses. Deployment "
                        "target: on-prem server with a mid-range GPU, near-real-time "
                        "(not hard real-time). Classes: forklift, pedestrian. "
                        "Propose an image budget and write the plan."
                    )
                )
            ]
        },
        config={"recursion_limit": 30},
    )

    print("=== FINAL MESSAGE ===")
    print(result["messages"][-1].content)

    print("\n=== class_budget.json ===")
    budget_file = _TEST_WORKSPACE / "class_budget.json"
    print(budget_file.read_text() if budget_file.exists() else "(not created)")

    print("=== plan.md ===")
    plan_file = _TEST_WORKSPACE / "plan.md"
    print(plan_file.read_text() if plan_file.exists() else "(not created)")


if __name__ == "__main__":
    asyncio.run(main())
