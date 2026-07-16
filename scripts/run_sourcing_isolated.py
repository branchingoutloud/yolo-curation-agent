"""Run the sourcing subagent in isolation against mocked planning output.

Levels (per sourcing-subagent-deep-dive.md "Testing the Agent in Isolation"):
  --mock : Level 1 — canned Roboflow/Kaggle search tools, no MCP servers.
           Needs only ANTHROPIC_API_KEY. Exercises budget-reading, schema
           conformance, status routing, and (with --round2) append behavior.
  (default) : Level 2 — real MCP tools from tools/mcp_clients.py. Needs
           ROBOFLOW_API_KEY and/or KAGGLE_MCP_URL / TAVILY_API_KEY.

Usage:
  python scripts/run_sourcing_isolated.py --mock --fresh   # round 1
  python scripts/run_sourcing_isolated.py --mock --round2  # append round
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import os  # noqa: E402  (after load_dotenv so .env wins over empty shell vars)

from deepagents import create_deep_agent  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from backends.project_backend import RUN_ARTIFACTS_DIR, project_backend  # noqa: E402
from subagents.sourcing import build_sourcing_agent  # noqa: E402
from tools.sources_schema import (  # noqa: E402
    coverage_summary,
    find_duplicate_dataset_ids,
    validate_sources,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"

ROUND1_PROMPT = """\
Search for public datasets for classes: car, truck, bus.
Per-class budget is in /workspace/class_budget.json - read it first.
Prefer already-annotated, permissively licensed (CC0/CC BY) sources in YOLO format.
Write every candidate to /workspace/sources.json and return a short gap summary.
"""

ROUND2_PROMPT = """\
'bus' is still thin: 0 annotated images found so far (target: 200).
Search specifically for bus detection datasets.
Append results to the existing /workspace/sources.json - do not drop or
rewrite the entries already there - and return an updated gap summary.
"""


# --- Level-1 mock search tools -------------------------------------------
# Canned results shaped like real search hits: car/truck well covered and
# annotated on Roboflow; bus only available raw on Kaggle in round 1, with an
# annotated bus set surfacing only on a targeted "bus" query (round 2). That
# split forces the agent through both the "available" and "needs_annotation"
# status paths and gives round 2 something genuinely new to append.

_MOCK_ROBOFLOW_HITS = [
    {
        "dataset": "traffic-detection/vehicles-openimages-4",
        "url": "https://universe.roboflow.com/traffic-detection/vehicles-openimages-4",
        "classes": {"car": 820, "truck": 410},
        "images": 1250,
        "annotation_format": "YOLO",
        "license": "CC BY 4.0",
        "annotated": True,
    },
    {
        "dataset": "highway-cams/truck-detection-2",
        "url": "https://universe.roboflow.com/highway-cams/truck-detection-2",
        "classes": {"truck": 260},
        "images": 300,
        "annotation_format": "YOLO",
        "license": "CC BY 4.0",
        "annotated": True,
    },
]

_MOCK_ROBOFLOW_BUS_HITS = [
    {
        "dataset": "city-transit/bus-detection-7",
        "url": "https://universe.roboflow.com/city-transit/bus-detection-7",
        "classes": {"bus": 240},
        "images": 240,
        "annotation_format": "YOLO",
        "license": "CC0",
        "annotated": True,
    },
]

_MOCK_KAGGLE_HITS = [
    {
        "dataset": "citycam/raw-bus-photos",
        "url": "https://www.kaggle.com/datasets/citycam/raw-bus-photos",
        "classes": {"bus": 0},
        "images": 300,
        "annotation_format": "none",
        "license": "CC0",
        "annotated": False,
    },
]


@tool
def mock_roboflow_search(query: str) -> str:
    """Search Roboflow Universe for annotated object-detection datasets
    matching the query. Returns a JSON list of hits."""
    hits = list(_MOCK_ROBOFLOW_HITS)
    if "bus" in query.lower():
        hits = _MOCK_ROBOFLOW_BUS_HITS
    return json.dumps(hits, indent=2)


@tool
def mock_kaggle_search(query: str) -> str:
    """Search Kaggle for image datasets matching the query. Returns a JSON
    list of hits (may be unannotated raw image collections)."""
    hits = [h for h in _MOCK_KAGGLE_HITS if any(c in query.lower() for c in h["classes"])]
    return json.dumps(hits if hits else _MOCK_KAGGLE_HITS, indent=2)


# --- harness ---------------------------------------------------------------


def seed_workspace(fresh: bool) -> Path:
    workspace = Path(RUN_ARTIFACTS_DIR)
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "class_budget.json", workspace / "class_budget.json")
    shutil.copy(FIXTURES / "plan.md", workspace / "plan.md")
    sources = workspace / "sources.json"
    if fresh and sources.exists():
        sources.unlink()
        print(f"[harness] removed stale {sources}")
    return workspace


def build_standalone_agent(use_mock: bool):
    """The sourcing SubAgent spec, run as a top-level agent — same system
    prompt, tools, and backend it gets under the orchestrator, minus the
    orchestrator."""
    if use_mock:
        spec = build_sourcing_agent([mock_roboflow_search], [mock_kaggle_search], [])
    else:
        from tools.mcp_clients import (
            get_kaggle_tools,
            get_roboflow_tools,
            get_web_search_tools,
        )

        spec = build_sourcing_agent(
            get_roboflow_tools(), get_kaggle_tools(), get_web_search_tools()
        )

    print(f"[harness] search tools: {[t.name for t in spec['tools']] or 'NONE'}")
    if not spec["tools"]:
        sys.exit(
            "[harness] no search tools loaded - set ROBOFLOW_API_KEY / "
            "KAGGLE_MCP_URL / TAVILY_API_KEY in .env, or use --mock"
        )

    model = os.environ.get("SOURCING_MODEL") or os.environ.get(
        "ORCHESTRATOR_MODEL", "anthropic:claude-sonnet-5"
    )
    return create_deep_agent(
        model=model,
        system_prompt=spec["system_prompt"],
        tools=spec["tools"],
        backend=project_backend,
    )


def verify_output(workspace: Path) -> int:
    sources_path = workspace / "sources.json"
    if not sources_path.exists():
        print("[verify] FAIL: sources.json was not written")
        return 1

    try:
        sources = json.loads(sources_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"[verify] FAIL: sources.json is not valid JSON: {exc}")
        return 1

    errors = validate_sources(sources)
    duplicates = find_duplicate_dataset_ids(sources if isinstance(sources, list) else [])
    for err in errors:
        print(f"[verify] schema error: {err}")
    for dup in duplicates:
        print(f"[verify] duplicate dataset_id: {dup}")

    budget = json.loads((workspace / "class_budget.json").read_text())
    print("\n[verify] coverage vs budget:")
    for name, row in coverage_summary(sources, budget).items():
        flag = "✓" if row["met"] else "✗"
        print(
            f"  {flag} {name}: {row['annotated']}/{row['target']} annotated"
            f" (raw: {row['raw']}, gap: {row['gap']})"
        )

    if errors or duplicates:
        print("\n[verify] FAIL")
        return 1
    print("\n[verify] PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="use canned search tools (Level 1)")
    parser.add_argument("--round2", action="store_true", help="targeted 'bus' follow-up (append test)")
    parser.add_argument("--fresh", action="store_true", help="delete any existing sources.json first")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("[harness] ANTHROPIC_API_KEY is not set - add it to .env")

    workspace = seed_workspace(fresh=args.fresh)
    agent = build_standalone_agent(use_mock=args.mock)

    prompt = ROUND2_PROMPT if args.round2 else ROUND1_PROMPT
    print(f"\n[harness] task prompt:\n{prompt}")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"recursion_limit": 100},
    )
    print(f"[harness] agent summary:\n{result['messages'][-1].content}\n")

    return verify_output(workspace)


if __name__ == "__main__":
    sys.exit(main())
