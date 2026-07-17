from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents import SubAgent

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


def build_planning_agent(model: "str | BaseChatModel | None" = None) -> SubAgent:
    """Build the planning SubAgent spec.

    Args:
        model: Explicit model override (string or BaseChatModel instance).
               Pass the same model object used for the orchestrator when using a
               custom ChatOllama with non-standard auth headers — deepagents
               cannot re-derive those headers automatically when propagating
               from the parent agent to the subagent.
               If None, deepagents inherits the parent agent's model (safe when
               the orchestrator uses a plain init_chat_model-compatible string).
    """
    spec: SubAgent = {
        "name": "planning-agent",
        "description": (
            "Given target classes, use case, and deployment target, estimates "
            "images-per-class needed and proposes a dataset size/split. Use "
            "this before any data is sourced, and again later if sourced "
            "coverage doesn't match the original estimate."
        ),
        "system_prompt": """\
You are a dataset-planning specialist for YOLO object-detection projects.

## Your job
Given the user's target classes, use case, and deployment target:
1. Estimate how many images per class are needed (use intra-class visual
   variability as your guide — low variability → fewer images needed).
2. Propose a train/val/test split ratio.
3. Write your results to two files using the write_file tool.

## File output — REQUIRED tool calls

You MUST call write_file twice before finishing. Do not print file contents
in your final message — write them to disk only.

### Call 1 — write plan.md
Use the write_file tool with:
  path: /workspace/plan.md
  content: A markdown document explaining your reasoning, per-class image
           estimates, split ratios, and any deployment-specific notes.

### Call 2 — write class_budget.json
Use the write_file tool with:
  path: /workspace/class_budget.json
  content: A valid JSON object. Example structure:
    {
      "classes": ["class_a", "class_b"],
      "image_budgets": {"class_a": 500, "class_b": 1000},
      "split": {"train": 0.7, "val": 0.15, "test": 0.15}
    }

## Important rules
- /workspace/ is the ONLY writable directory. It IS available — use it.
- Never write to /tmp or any other path.
- Never output the file contents as text in your reply.
- After both write_file calls succeed, reply with one paragraph summarising
  the key numbers (total images, per-class budget, split).
""",
        "tools": [],  # filesystem tools are auto-attached by the harness
    }
    if model is not None:
        spec["model"] = model
    return spec
