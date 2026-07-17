"""Exercise sourcing-agent end-to-end against a real Roboflow Universe search
(read-only discovery, not a fetch - much lighter than dataset-agent's test,
so should stay well under free-tier token limits even on Groq).

Requires ROBOFLOW_API_KEY in .env to actually search; without it, sourcing-
agent has zero Roboflow tools and will say so rather than fabricate results -
that's expected, not a bug.

Usage:
    python scripts/test_sourcing_agent.py
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
from subagents.sourcing import build_sourcing_agent  # noqa: E402
from tools.mcp_clients import get_kaggle_tools, get_roboflow_tools, get_web_search_tools  # noqa: E402
from tools.model_builder import build_model  # noqa: E402


async def main(roboflow_tools: list, kaggle_tools: list, web_search_tools: list):
    if _TEST_WORKSPACE.exists():
        import shutil

        shutil.rmtree(_TEST_WORKSPACE)
    _TEST_WORKSPACE.mkdir(parents=True)

    print(f"Roboflow tools available: {len(roboflow_tools)} {'(set ROBOFLOW_API_KEY in .env to enable)' if not roboflow_tools else ''}")
    print(f"Web search tools available: {len(web_search_tools)}\n")

    spec = build_sourcing_agent(roboflow_tools, kaggle_tools, web_search_tools)
    print(f"sourcing-agent's actual tool list: {[getattr(t, 'name', str(t)) for t in spec['tools']]}\n")

    # sourcing-agent has no model override of its own - mirror the real
    # orchestrator's ORCHESTRATOR_MODEL fallback (see agent.py) via
    # build_model (not a raw string) so this test gets the same Ollama Cloud
    # base_url/API-key/concurrency-lock wiring the real agent gets.
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
                        "Find datasets for detecting 'forklift' in warehouse images. "
                        "Catalog whatever you find in sources.json."
                    )
                )
            ]
        },
        config={"recursion_limit": 30},
    )

    print("=== FINAL MESSAGE ===")
    print(result["messages"][-1].content)

    print("\n=== sources.json ===")
    sources_file = _TEST_WORKSPACE / "sources.json"
    print(sources_file.read_text() if sources_file.exists() else "(not created)")


if __name__ == "__main__":
    _roboflow_tools = get_roboflow_tools()
    _kaggle_tools = get_kaggle_tools()
    _web_search_tools = get_web_search_tools()
    asyncio.run(main(_roboflow_tools, _kaggle_tools, _web_search_tools))
