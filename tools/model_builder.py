"""Shared model-construction helper so the orchestrator and every subagent
build Ollama Cloud models the same way, instead of each duplicating the
base_url/API-key wiring (this used to live only in subagents/dataset.py's
`_dataset_agent_model`).

Ollama Cloud (https://ollama.com) runs large hosted models (gpt-oss:120b,
gpt-oss:20b, etc.) over the same HTTP API a local `ollama serve` exposes -
the only differences are the host (https://ollama.com instead of
http://localhost:11434) and a Bearer API key. Set OLLAMA_API_KEY to switch
an "ollama:<model>" spec from local/unauthenticated to cloud/authenticated;
OLLAMA_BASE_URL only needs to be set explicitly to override either default
(e.g. pointing at the office GPU's own network address instead of either
localhost or the cloud endpoint).

Non-"ollama:" specs (anthropic:, groq:, huggingface: once langchain-
huggingface + a real HUGGINGFACEHUB_API_TOKEN are added, etc.) pass straight
through to init_chat_model unchanged - huggingface is not wired with a
default model/token since none was given, but any subagent's *_AGENT_MODEL
override can point at one the moment a real token exists, no code change
needed.
"""

import asyncio
import os
import threading
import time

from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
from ollama import ResponseError

# The free-tier Ollama Cloud key this project is using allows only ONE
# in-flight cloud call at a time - a second concurrent call errors rather
# than queueing. Subagents can genuinely run concurrently (the orchestrator
# is allowed to issue parallel `task` calls in one turn), so every cloud
# model built by build_model() must serialize through the SAME process-wide
# lock - not one lock per model instance, since the limit is per API key/
# account, not per object. Local Ollama (no OLLAMA_API_KEY - e.g. tomorrow's
# office GPU) has no such constraint, so only the cloud path below is
# wrapped; local calls run unthrottled.
_CLOUD_CALL_LOCK = threading.Lock()
_CLOUD_CALL_ASYNC_LOCK = asyncio.Lock()

# Observed live (twice): Ollama Cloud occasionally returns a transient
# `ollama._types.ResponseError` with status 500 on an otherwise-valid
# request - not something our code can prevent, but retrying once or twice
# is worth it since one blip otherwise kills the entire orchestrator run
# (an uncaught exception from the model node crashes the graph same as an
# uncaught exception from a tool - see the ValueError-crash story in
# tools/dataset_builder.py's history for the same underlying lesson).
# Client-error status codes (4xx - bad model name, bad auth) are NOT
# retried; retrying those just wastes the retry budget on something that
# will never succeed.
_RETRYABLE_STATUS_CODES = {500, 502, 503, 504}
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = 2


def _is_retryable(exc: Exception) -> bool:
    return isinstance(exc, ResponseError) and exc.status_code in _RETRYABLE_STATUS_CODES


class _ThrottledCloudChatOllama(ChatOllama):
    """ChatOllama that serializes every call behind the shared cloud lock and
    retries transient 5xx errors.

    Covers all four entry points BaseChatModel funnels invoke/ainvoke/stream/
    astream through. Sync and async calls are only serialized against calls
    of the same kind (separate locks) - acceptable here since real usage is
    either the fully-async langgraph server or a single standalone test
    script, never both against the same key at once.

    The streaming variants only retry if nothing has been yielded yet in the
    failing attempt - once a caller has received a partial chunk there's no
    safe way to "undo" it, so a mid-stream failure past the first chunk still
    raises rather than risking duplicated/garbled output.
    """

    def _generate(self, *args, **kwargs):
        with _CLOUD_CALL_LOCK:
            for attempt in range(_MAX_ATTEMPTS):
                try:
                    return super()._generate(*args, **kwargs)
                except Exception as exc:
                    if attempt == _MAX_ATTEMPTS - 1 or not _is_retryable(exc):
                        raise
                    time.sleep(_BACKOFF_SECONDS * (attempt + 1))

    def _stream(self, *args, **kwargs):
        with _CLOUD_CALL_LOCK:
            for attempt in range(_MAX_ATTEMPTS):
                yielded_any = False
                try:
                    for chunk in super()._stream(*args, **kwargs):
                        yielded_any = True
                        yield chunk
                    return
                except Exception as exc:
                    if yielded_any or attempt == _MAX_ATTEMPTS - 1 or not _is_retryable(exc):
                        raise
                    time.sleep(_BACKOFF_SECONDS * (attempt + 1))

    async def _agenerate(self, *args, **kwargs):
        async with _CLOUD_CALL_ASYNC_LOCK:
            for attempt in range(_MAX_ATTEMPTS):
                try:
                    return await super()._agenerate(*args, **kwargs)
                except Exception as exc:
                    if attempt == _MAX_ATTEMPTS - 1 or not _is_retryable(exc):
                        raise
                    await asyncio.sleep(_BACKOFF_SECONDS * (attempt + 1))

    async def _astream(self, *args, **kwargs):
        async with _CLOUD_CALL_ASYNC_LOCK:
            for attempt in range(_MAX_ATTEMPTS):
                yielded_any = False
                try:
                    async for chunk in super()._astream(*args, **kwargs):
                        yielded_any = True
                        yield chunk
                    return
                except Exception as exc:
                    if yielded_any or attempt == _MAX_ATTEMPTS - 1 or not _is_retryable(exc):
                        raise
                    await asyncio.sleep(_BACKOFF_SECONDS * (attempt + 1))


# Confirmed against the installed langchain-ollama version (chat_models.py):
# ChatOllama forwards `client_kwargs` straight to `ollama.Client(host=...,
# **client_kwargs)`, and `client_kwargs["headers"]` is merged into the
# underlying httpx client's headers - this is the supported way to attach
# Ollama Cloud's Bearer-token auth. Verified end-to-end against a real
# Ollama Cloud key/account: cloud model tags do NOT need a "-cloud" suffix -
# GET {base_url}/api/tags lists them bare (e.g. "gpt-oss:120b"), and a real
# /api/chat call against that bare tag succeeds. If a future cloud call 404s
# on a model name, check the account's own /api/tags output first (the
# catalog changes over time) rather than assuming a suffix is needed.


def build_model(model_spec: str | None):
    """Resolve a "provider:model" spec into a model object, or None.

    Returns None when model_spec is falsy, so callers can omit `SubAgent`'s
    optional "model" key entirely - deepagents' own subagent-building loop
    (`spec.get("model", model)` in graph.py) then falls back to whatever the
    orchestrator's model resolved to, identically across every subagent.
    Also falls back to None if construction fails for any reason (missing
    provider package, bad model string, unreachable host), following this
    repo's existing lazy/defensive pattern for optional integrations (see
    tools/mcp_clients.py) rather than crashing agent.py at import time.
    """
    if not model_spec:
        return None

    if model_spec.startswith("ollama:"):
        model_name = model_spec.removeprefix("ollama:")
        api_key = os.environ.get("OLLAMA_API_KEY")
        base_url = os.environ.get(
            "OLLAMA_BASE_URL", "https://ollama.com" if api_key else "http://localhost:11434"
        )
        try:
            if api_key:
                return _ThrottledCloudChatOllama(
                    model=model_name,
                    base_url=base_url,
                    client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}},
                )
            return ChatOllama(model=model_name, base_url=base_url)
        except Exception as exc:  # noqa: BLE001 - degrade, don't crash startup
            print(f"[model_builder] failed to init {model_spec!r} ({exc}); falling back to caller's default.")
            return None

    try:
        return init_chat_model(model_spec)
    except Exception as exc:  # noqa: BLE001 - degrade, don't crash startup
        print(f"[model_builder] failed to init {model_spec!r} ({exc}); falling back to caller's default.")
        return None
