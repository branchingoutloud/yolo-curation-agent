"""Orchestrator wiring for the YOLO curation & training deep agent.

The exported `agent` is what langgraph.json points at (./agent.py:agent).
See yolo-deep-agent-architecture.md for the full design rationale - this
file is the §3 code sample wired up to the actual subagents/tools/backends
modules in this repo instead of inline placeholders.
"""

from deepagents import create_deep_agent

from backends.project_backend import project_backend
from subagents.annotation import build_annotation_agent
from subagents.dataset import build_dataset_agent
from subagents.eval import build_eval_agent
from subagents.planning import build_planning_agent
from subagents.sourcing import build_sourcing_agent
from subagents.training import build_training_agent
from tools.approval import request_approval
from tools.mcp_clients import get_kaggle_tools, get_roboflow_tools, get_web_search_tools
from tools.models import build_default_model
from tools.zero_shot_annotate import zero_shot_annotate

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

# Virtual backend path, not a host filesystem path — SkillsMiddleware sources
# are paths *in the backend*; project_backend routes "/skills/" to a
# FilesystemBackend rooted at the real skills/ directory (see
# backends/project_backend.py).
SKILLS_DIR = "/skills"


def build_agent():
    roboflow_tools = get_roboflow_tools()
    kaggle_tools = get_kaggle_tools()
    web_search_tools = get_web_search_tools()

    # A constructed model *object* (not a bare "provider:model" string) is
    # required here so Ollama Cloud's custom base_url/auth header survive
    # deepagents' per-subagent model inheritance — see tools/models.py.
    model = build_default_model()

    subagents = [
        build_planning_agent(),
        build_sourcing_agent(roboflow_tools, kaggle_tools, web_search_tools),
        build_annotation_agent(roboflow_tools, [zero_shot_annotate]),
        build_dataset_agent(),
        build_training_agent(),
        build_eval_agent(),
    ]

    return create_deep_agent(
        model=model,
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
