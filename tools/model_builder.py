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

# Ollama Cloud (especially the free tier) intermittently returns 5xx on an
# otherwise-valid request - documented in CLAUDE.md. Retry a few times with
# exponential backoff so one transient blip doesn't kill a whole graph run.
# Overridable via env (set OLLAMA_MAX_RETRIES=0 to disable).
_RETRY_STATUS = {500, 502, 503, 504}
_MAX_RETRIES = int(os.environ.get("OLLAMA_MAX_RETRIES", "2"))
_RETRY_BASE_DELAY = float(os.environ.get("OLLAMA_RETRY_BACKOFF", "2.0"))

try:  # recent ollama re-exports ResponseError at the top level
    from ollama import ResponseError as _ResponseError
except Exception:  # noqa: BLE001 - fall back to the private path, else disable status checks
    try:
        from ollama._types import ResponseError as _ResponseError
    except Exception:  # noqa: BLE001
        _ResponseError = None

# httpx transient transport failures (matched by class name to avoid a hard
# httpx import here) - connection resets, read timeouts, etc.
_TRANSIENT_EXC_NAMES = {
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "PoolTimeout",
    "WriteError",
}


def _is_retryable(exc: Exception) -> bool:
    """True for transient Ollama Cloud failures worth retrying."""
    if _ResponseError is not None and isinstance(exc, _ResponseError):
        return getattr(exc, "status_code", None) in _RETRY_STATUS
    return type(exc).__name__ in _TRANSIENT_EXC_NAMES


def _log_retry(exc: Exception, attempt: int, delay: float, *, pre_stream: bool = False) -> None:
    status = getattr(exc, "status_code", "")
    where = " (pre-stream)" if pre_stream else ""
    print(
        f"[model_builder] transient {type(exc).__name__} {status} from Ollama Cloud{where}; "
        f"retry {attempt + 1}/{_MAX_RETRIES} in {delay:.0f}s"
    )


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
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    return super()._generate(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - retry transient, re-raise the rest
                    if attempt >= _MAX_RETRIES or not _is_retryable(exc):
                        raise
                    delay = _RETRY_BASE_DELAY * (2**attempt)
                    _log_retry(exc, attempt, delay)
                    time.sleep(delay)

    def _stream(self, *args, **kwargs):
        with _CLOUD_CALL_LOCK:
            for attempt in range(_MAX_RETRIES + 1):
                started = False
                try:
                    for chunk in super()._stream(*args, **kwargs):
                        started = True
                        yield chunk
                    return
                except Exception as exc:  # noqa: BLE001
                    # Only safe to retry before the first chunk - restarting
                    # after emitting content would duplicate output.
                    if started or attempt >= _MAX_RETRIES or not _is_retryable(exc):
                        raise
                    delay = _RETRY_BASE_DELAY * (2**attempt)
                    _log_retry(exc, attempt, delay, pre_stream=True)
                    time.sleep(delay)

    async def _agenerate(self, *args, **kwargs):
        async with _CLOUD_CALL_ASYNC_LOCK:
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    return await super()._agenerate(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    if attempt >= _MAX_RETRIES or not _is_retryable(exc):
                        raise
                    delay = _RETRY_BASE_DELAY * (2**attempt)
                    _log_retry(exc, attempt, delay)
                    await asyncio.sleep(delay)

    async def _astream(self, *args, **kwargs):
        async with _CLOUD_CALL_ASYNC_LOCK:
            for attempt in range(_MAX_RETRIES + 1):
                started = False
                try:
                    async for chunk in super()._astream(*args, **kwargs):
                        started = True
                        yield chunk
                    return
                except Exception as exc:  # noqa: BLE001
                    if started or attempt >= _MAX_RETRIES or not _is_retryable(exc):
                        raise
                    delay = _RETRY_BASE_DELAY * (2**attempt)
                    _log_retry(exc, attempt, delay, pre_stream=True)
                    await asyncio.sleep(delay)


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
        # reasoning=True: without it, langchain_ollama silently discards gpt-oss's
        # "thinking" tokens instead of merging/surfacing them - if the model exhausts
        # its output budget mid-thought (e.g. a long tool-heavy turn), that produces a
        # totally empty AIMessage with no error, which a ReAct loop reads as "done".
        try:
            if api_key:
                return _ThrottledCloudChatOllama(
                    model=model_name,
                    base_url=base_url,
                    client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}},
                    reasoning=True,
                )
            return ChatOllama(model=model_name, base_url=base_url, reasoning=True)
        except Exception as exc:  # noqa: BLE001 - degrade, don't crash startup
            print(f"[model_builder] failed to init {model_spec!r} ({exc}); falling back to caller's default.")
            return None

    try:
        return init_chat_model(model_spec)
    except Exception as exc:  # noqa: BLE001 - degrade, don't crash startup
        print(f"[model_builder] failed to init {model_spec!r} ({exc}); falling back to caller's default.")
        return None
