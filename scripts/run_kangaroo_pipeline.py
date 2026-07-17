"""Run the full dataset curation and training pipeline sequentially for a specific query.

This script manually orchestrates the subagents (Planning -> Sourcing -> Dataset)
in sequence. It allows you to check the output of each subagent and trace the exact
flow of artifacts (class_budget.json, sources.json, dataset/) for the specific
"kangaroo" query.

Usage:
    python scripts/run_kangaroo_pipeline.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Fix console encoding for cloud models that emit unicode
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Isolate the workspace for this run so it doesn't overwrite real run_artifacts
_WORKSPACE = Path(__file__).resolve().parent.parent / "run_artifacts_kangaroo"
os.environ["RUN_ARTIFACTS_DIR"] = str(_WORKSPACE)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage

from backends.project_backend import project_backend
from subagents.planning import build_planning_agent
from subagents.sourcing import build_sourcing_agent
from subagents.dataset import build_dataset_agent
# You can also import build_training_agent and build_eval_agent if you want to run the full training locally
# from subagents.training import build_training_agent
# from subagents.eval import build_eval_agent

from tools.mcp_clients import get_kaggle_tools, get_roboflow_tools, get_web_search_tools
from tools.model_builder import build_model

async def main():
    if _WORKSPACE.exists():
        import shutil
        shutil.rmtree(_WORKSPACE)
    _WORKSPACE.mkdir(parents=True)
    
    print(f"--- Workspace created at: {_WORKSPACE} ---\n")

    # Resolve the model (mirroring the orchestrator's fallback logic)
    model_spec = os.environ.get("ORCHESTRATOR_MODEL", "groq:openai/gpt-oss-120b")
    model = build_model(model_spec) or model_spec
    print(f"Using Model: {model_spec}\n")

    # --- 1. Planning Subagent ---
    print("========================================")
    print(" 1. RUNNING PLANNING AGENT")
    print("========================================")
    plan_spec = build_planning_agent()
    plan_agent = create_deep_agent(
        model=model, system_prompt=plan_spec["system_prompt"], tools=plan_spec["tools"], backend=project_backend
    )
    
    plan_result = await plan_agent.ainvoke({
        "messages": [HumanMessage(content=(
            "I want to train a YOLO model for identifying a kangaroo. "
            "Please create a plan and class budget for this use case."
        ))]
    }, config={"recursion_limit": 30})
    
    print("\n=== Planning Agent Output ===")
    print(plan_result["messages"][-1].content)
    
    budget_file = _WORKSPACE / "class_budget.json"
    print(f"\nCreated class_budget.json: {budget_file.exists()}")


    # --- 2. Sourcing Subagent ---
    print("\n\n========================================")
    print(" 2. RUNNING SOURCING AGENT")
    print("========================================")
    roboflow_tools = get_roboflow_tools()
    kaggle_tools = get_kaggle_tools()
    web_search_tools = get_web_search_tools()
    
    source_spec = build_sourcing_agent(roboflow_tools, kaggle_tools, web_search_tools)
    source_agent = create_deep_agent(
        model=model, system_prompt=source_spec["system_prompt"], tools=source_spec["tools"], backend=project_backend
    )
    
    source_result = await source_agent.ainvoke({
        "messages": [HumanMessage(content=(
            "Read the class_budget.json and find dataset sources for a kangaroo. "
            "Write the candidate datasets to sources.json and provide a gap summary."
        ))]
    }, config={"recursion_limit": 50})
    
    print("\n=== Sourcing Agent Output ===")
    print(source_result["messages"][-1].content)
    
    sources_file = _WORKSPACE / "sources.json"
    print(f"\nCreated sources.json: {sources_file.exists()}")
    if sources_file.exists():
        print(sources_file.read_text()[:300] + "...\n")


    # --- 3. Dataset Subagent ---
    print("\n========================================")
    print(" 3. RUNNING DATASET AGENT")
    print("========================================")
    dataset_spec = build_dataset_agent(roboflow_tools, kaggle_tools)
    dataset_agent = create_deep_agent(
        model=model, system_prompt=dataset_spec["system_prompt"], tools=dataset_spec["tools"], backend=project_backend
    )
    
    dataset_result = await dataset_agent.ainvoke({
        "messages": [HumanMessage(content=(
            "Read sources.json and download/merge the datasets into a finalized "
            "YOLO dataset format in the workspace."
        ))]
    }, config={"recursion_limit": 50})
    
    print("\n=== Dataset Agent Output ===")
    print(dataset_result["messages"][-1].content)
    
    dataset_dir = _WORKSPACE / "dataset"
    print(f"\nCreated dataset/ directory: {dataset_dir.exists()}")
    
    print("\n--- Pipeline Execution Complete ---")

if __name__ == "__main__":
    asyncio.run(main())
