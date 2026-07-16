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
    return f"Approval requested for {stage}: {summary}"
