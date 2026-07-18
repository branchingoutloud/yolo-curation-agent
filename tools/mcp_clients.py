"""Roboflow + Kaggle MCP tool loading, plus an optional web-search tool.

Kept lazy and defensive on purpose: `langgraph dev` should still boot even if
a server's credentials aren't set yet (missing env var -> that tool list is
just empty, not a crash) or if a server is briefly unreachable (§13 risk:
"Roboflow/Kaggle rate limits or auth hiccups live"). Each get_*_tools() call
prints a warning and degrades to [] rather than raising, so agent.py's
startup never depends on every integration being configured.

Roboflow's MCP server (mcp.roboflow.com) does not implement the optional
Streamable HTTP affordances the MCP SDK's client tries opportunistically:
GET (server->client push stream) and DELETE (explicit session termination)
both come back 405, confirmed live. This is spec-legal, not a bug on
Roboflow's end - the transport spec explicitly allows a 405 for both, and
the next protocol revision removes GET-stream/session-termination from the
transport entirely. Two consequences handled here:
  1. `terminate_on_close: False` on the roboflow connection below skips the
     doomed DELETE at the end of every tool-call session instead of sending
     it and swallowing a 405 every time (langchain_mcp_adapters opens a new
     session per tool call, so this would otherwise happen constantly).
  2. The `mcp.client.streamable_http` logger is dropped to WARNING at import
     time (below) because the GET-stream retry loop logs its "reconnecting"
     message at INFO by default - there is no client-side flag to stop the
     GET attempt itself (only DELETE has an opt-out), so quieting the logger
     is the only lever available short of patching the installed SDK.
  Neither changes functional behavior - every real tool call already
  succeeds over POST regardless (verified live: sourcing-agent's
  universe_search/append_sources run, dataset-agent's fork/export chain).
  Kaggle hasn't been confirmed to share this behavior, so it's left on
  defaults rather than assumed.
"""

import asyncio
import logging
import os
from functools import lru_cache

from langchain_mcp_adapters.client import MultiServerMCPClient

logging.getLogger("mcp.client.streamable_http").setLevel(logging.WARNING)


def _build_mcp_client() -> MultiServerMCPClient | None:
    servers = {}

    roboflow_key = os.environ.get("ROBOFLOW_API_KEY")
    if roboflow_key:
        servers["roboflow"] = {
            "url": os.environ.get("ROBOFLOW_MCP_URL", "https://mcp.roboflow.com/mcp"),
            "transport": "streamable_http",
            "headers": {"Authorization": f"Bearer {roboflow_key}"},
            # Roboflow's MCP server 405s session-termination DELETEs (see
            # module docstring) - skip sending one rather than doing a
            # doomed round trip on every single tool call.
            "terminate_on_close": False,
        }

    kaggle_url = os.environ.get("KAGGLE_MCP_URL")
    if kaggle_url:
        kaggle_key = os.environ.get("KAGGLE_API_KEY")
        servers["kaggle"] = {
            "url": kaggle_url,
            "transport": "streamable_http",
            **({"headers": {"Authorization": f"Bearer {kaggle_key}"}} if kaggle_key else {}),
        }

    if not servers:
        return None
    return MultiServerMCPClient(servers)


@lru_cache(maxsize=1)
def _mcp_client() -> MultiServerMCPClient | None:
    return _build_mcp_client()


def _get_tools_sync(server_name: str) -> list:
    client = _mcp_client()
    if client is None or server_name not in client.connections:
        # Either no server has credentials at all, or this specific one
        # wasn't registered (e.g. KAGGLE_MCP_URL unset while ROBOFLOW_API_KEY
        # is set) - both are normal degraded-config states, not errors worth
        # logging.
        return []
    try:
        return asyncio.run(client.get_tools(server_name=server_name))
    except Exception as exc:  # startup diagnostics only, never fatal
        print(f"[mcp_clients] failed to load '{server_name}' MCP tools: {exc}")
        return []


def get_roboflow_tools() -> list:
    return _get_tools_sync("roboflow")


def get_kaggle_tools() -> list:
    return _get_tools_sync("kaggle")


def get_web_search_tools() -> list:
    """Optional web_search tool for sourcing-agent (papers, niche dataset
    pointers), gated on TAVILY_API_KEY. Returns [] if not configured -
    sourcing-agent still works on Roboflow/Kaggle alone.
    """
    if not os.environ.get("TAVILY_API_KEY"):
        return []
    try:
        from langchain_tavily import TavilySearch

        return [TavilySearch(max_results=5)]
    except Exception as exc:  # pragma: no cover - startup diagnostics only
        print(f"[mcp_clients] failed to load web_search tool: {exc}")
        return []
