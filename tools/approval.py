"""The orchestrator's only non-delegation tool - the three HITL gates.

Gating this dedicated tool (rather than gating `task` itself) is deliberate:
the orchestrator may call sourcing-agent/annotation-agent an unknown number
of times, so an interrupt on every `task` call would pause the graph far
more often than the three moments that actually need a human. See §9 of the
architecture doc.
"""

from langchain_core.tools import tool


@tool
def request_approval(stage: str, summary: str) -> str:
    """Present a plan/decision to the user and pause for approval before
    proceeding. Call this exactly at: (1) after planning, before any data is
    sourced; (2) after proposing a model size, before training starts; (3)
    after eval results, before an iteration-2 sourcing pass. `summary` must
    be plain, readable text (not a JSON blob) - it is what renders in the
    approval card the user sees.
    """
    # This function body only ever runs for real on an "approve"/"edit" HITL
    # decision - langchain's HumanInTheLoopMiddleware short-circuits "reject"
    # with a synthetic ToolMessage instead of calling this tool at all. So the
    # return here must say the request was actually approved, not just that
    # it was made - a neutral echo of the request (the old behavior) left the
    # model unable to tell approval had happened and made it re-ask instead
    # of proceeding.
    return f"User approved this request for stage '{stage}'. Proceed: {summary}"
