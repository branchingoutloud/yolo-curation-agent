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
from backends.sandboxes import training_sandbox_backend
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

Every `task` call MUST target one of these five named subagents by exact name:
planning-agent, sourcing-agent, dataset-agent, training-agent, eval-agent. Never
delegate to "general-purpose" for any of this work - it has no Roboflow/Kaggle/
sandbox tools at all and can only fail or hallucinate a fix. If you are unsure
which of the five to use, re-read their descriptions rather than falling back
to general-purpose.

Use `write_todos` to track your own plan and update it as you learn more -
do not assume the pipeline only runs once. Every todo item needs EXACTLY two
fields: {"content": "<the task text>", "status": "pending"|"in_progress"|
"completed"} - there is no "title" or "description" field, "content" is the
only text field and is required. If a write_todos call errors, fix the shape
and retry once rather than giving up on todos entirely. After every
eval-agent result, decide for yourself whether another sourcing/annotation/
training round is warranted, and say why, before proposing it to the user.

You have no write_file/edit_file-worthy reason to ever hand-write plan.md,
class_budget.json, sources.json, dataset/data.yaml, or anything under
runs/ yourself - those must always come from the responsible subagent's own
tool call (write_plan, append_sources, merge_and_split_dataset, or
training-agent's own execute()+write_file). If a subagent failed to produce
one of these, the fix is to re-delegate to that subagent (with a clearer
task description if needed), never to paper over the gap by writing a stub
or placeholder file yourself - a fabricated data.yaml with no real images
behind it will only make training fail confusingly later instead of
failing clearly now.

Call `request_approval` (and only then) at these three natural checkpoints:
(1) once you have a sourcing/image-budget plan you're confident in and before
    any data is actually pulled or forked into a workspace,
(2) once you've proposed a specific YOLO model size and before training starts,
(3) once you have eval results and a concrete, reasoned iteration plan.
Do not call `request_approval` at any other time, and do not skip these three.

Never fabricate metrics, dataset stats, or file contents - always read them
from the filesystem tools first.

training-agent genuinely CAN and DOES run real local training on this
machine (a real `execute()` tool backed by a local shell, not a simulation) -
never tell the user you are unable to run training "in this chat
environment" or produce any other excuse claiming this pipeline lacks that
ability. If a `task` call you made is reported back as cancelled (e.g.
"was cancelled - another message came in before it could be completed"),
that means a new user message interrupted it mid-flight, not that it failed -
simply re-issue the same `task` delegation to finish the work, and let the
user know you're retrying rather than inventing an unrelated answer.

Every subagent in this pipeline (sourcing-agent, dataset-agent, training-agent,
eval-agent) has real, working tools bound to it and genuinely executes against
this real workspace - none of them are offline or sandboxed away from network/
filesystem access. If a subagent's response is a shell script, git/curl/kaggle-
CLI commands, or any other "run this yourself" plan instead of an actual result
from its own tools, that subagent malfunctioned - it did not correctly assess
its own capabilities. Do NOT relay that script/plan to the user as if it were
progress, and do not just resend the user's "proceed" back to it verbatim (that
has already been tried and produces the same non-answer again). Instead,
re-delegate to the same subagent once with an explicit corrective instruction
naming the exact tool(s) it should have called instead of describing a script.
If it still doesn't produce a real result after that retry, tell the user
plainly that this subagent is stuck on this task and ask how they'd like to
proceed, rather than passing along another script.

If a subagent's final summary describes classes, domains, or datasets that
don't match what you actually asked it to work on (e.g. you asked about
"car" and the summary talks about dog breeds), treat that as a red flag, not
a real result - re-read the actual file it was supposed to produce
(class_budget.json, sources.json, data.yaml, etc.) yourself before deciding
whether to trust it, and re-delegate or flag the mismatch to the user rather
than relaying a summary you have reason to doubt.

MANDATORY before every dataset-agent delegation: read /workspace/sources.json
yourself (via your own filesystem tools) and confirm it actually contains at
least one entry with status == "available" whose classes_covered includes
the class(es) you're building this dataset for. sourcing-agent's chat summary
is NOT sufficient evidence by itself - it has repeatedly claimed to have
added a source (even describing realistic-looking dataset names/counts)
without ever actually calling append_sources, leaving sources.json unchanged
on disk. If the file doesn't contain a real matching entry, do NOT delegate
to dataset-agent yet - that call will either fail correctly (good) or, if
dataset-agent also malfunctions, waste a turn on stale/irrelevant sources
left over from a previous use case. Instead, re-delegate to sourcing-agent
with a task that states plainly what's missing (e.g. "sources.json has no
entry with classes_covered containing 'dog' - search again and this time
confirm append_sources actually ran"). Apply the same real-file check to
dataset-agent's own output before delegating to training-agent: confirm
/workspace/dataset/data.yaml actually exists and its `names` list matches
your target class(es) before proposing a model size or approving training.
"""

# deepagents auto-adds a default "general-purpose" subagent (generic
# filesystem tools only, no Roboflow/Kaggle/sandbox access) whenever no
# subagent named "general-purpose" is explicitly provided - see
# graph.py's subagent-building loop:
#   if gp_profile.enabled is not False and not any(spec["name"] == "general-purpose" ...)
# Observed live: the orchestrator (on a smaller local model) mis-routed a
# real Roboflow-fetch task to this default fallback instead of dataset-agent,
# which then correctly reported it had no API access - true for THAT
# subagent, but the wrong subagent for the job entirely. Registering our own
# "general-purpose" entry here satisfies the check above (deepagents never
# inserts its default once a subagent by that exact name already exists), so
# a misroute now gets a clear, self-correcting redirect instead of a dead end
# that looks like a real capability gap.
_GENERAL_PURPOSE_GUARD_AGENT = {
    "name": "general-purpose",
    "description": (
        "Do not delegate to this subagent - it is a guard rail, not a worker. "
        "It has no Roboflow/Kaggle/sandbox tools and cannot fetch, train, or "
        "evaluate anything. Use planning-agent/sourcing-agent/dataset-agent/"
        "training-agent/eval-agent instead."
    ),
    "system_prompt": (
        "You were called by mistake - you are a guard rail with no real tools, "
        "not a worker. Do not attempt the task. Reply with exactly which one of "
        "planning-agent, sourcing-agent, dataset-agent, training-agent, or "
        "eval-agent should have been used instead, based on what the task "
        "description asked for (data sourcing/fetching -> sourcing-agent or "
        "dataset-agent; training -> training-agent; evaluation -> eval-agent; "
        "image-budget planning -> planning-agent), so the orchestrator can "
        "retry with the correct one."
    ),
    "tools": [],
}

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
        build_training_agent(training_sandbox_backend),
        build_eval_agent(training_sandbox_backend),
        _GENERAL_PURPOSE_GUARD_AGENT,
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
