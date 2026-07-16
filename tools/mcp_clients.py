"""Roboflow + Kaggle MCP tool loading, plus an optional web-search tool.

Kept lazy and defensive on purpose: `langgraph dev` should still boot even if
a server's credentials aren't set yet (missing env var -> that tool list is
just empty, not a crash) or if a server is briefly unreachable (§13 risk:
"Roboflow/Kaggle rate limits or auth hiccups live"). Each get_*_tools() call
prints a warning and degrades to [] rather than raising, so agent.py's
startup never depends on every integration being configured.
"""

import asyncio
import os
from functools import lru_cache

from langchain_mcp_adapters.client import MultiServerMCPClient


def _build_mcp_client() -> MultiServerMCPClient | None:
    servers = {}

    roboflow_key = os.environ.get("ROBOFLOW_API_KEY")
    if roboflow_key:
        servers["roboflow"] = {
            "url": os.environ.get("ROBOFLOW_MCP_URL", "https://mcp.roboflow.com"),
            "transport": "streamable_http",
            "headers": {"Authorization": f"Bearer {roboflow_key}"},
        }

    kaggle_url = os.environ.get("KAGGLE_MCP_URL")
    if kaggle_url:
        servers["kaggle"] = {"url": kaggle_url, "transport": "streamable_http"}

    if not servers:
        return None
    return MultiServerMCPClient(servers)


@lru_cache(maxsize=1)
def _mcp_client() -> MultiServerMCPClient | None:
    return _build_mcp_client()


def _get_tools_sync(server_name: str) -> list:
    client = _mcp_client()
    if client is None:
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
