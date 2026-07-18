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
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_ollama import ChatOllama
from ollama import ResponseError

# The free-tier Ollama Cloud key this project is using allows only ONE
# in-flight cloud call at a time - a second concurrent call errors rather
# than queueing. Subagents can genuinely run concurrently (the orchestrator
# is allowed to issue parallel `task` calls in one turn), so every cloud
# model built by build_model() must serialize through the SAME process-wide
# lock - not one lock per model instance, since the limit is per API key/
# account, not per object. Local Ollama has no such constraint (it's Ollama's
# own job to queue concurrent requests against one GPU), so only the cloud
# path adds this lock - both local and cloud get the retry behavior below.
_CLOUD_CALL_LOCK = threading.Lock()
_CLOUD_CALL_ASYNC_LOCK = asyncio.Lock()

# Observed live: Ollama Cloud occasionally returns a transient
# `ollama._types.ResponseError` with status 500 on an otherwise-valid
# request; a local model has separately produced a malformed/truncated
# tool-call JSON that Ollama's own client couldn't parse (status_code -1 -
# `ResponseError`'s default when raised client-side, not from a real HTTP
# response - see ollama/_client.py's bare `raise ResponseError(err)`).
# Neither is something our code can prevent, but retrying once or twice is
# worth it either way since one blip otherwise kills the entire orchestrator
# run (an uncaught exception from the model node crashes the graph same as
# an uncaught exception from a tool - see the ValueError-crash story in
# tools/dataset_builder.py's history for the same underlying lesson). A
# malformed-JSON generation is exactly the kind of thing a fresh sampling
# attempt is likely to just not repeat.
# Client-error status codes (4xx - bad model name, bad auth) are NOT
# retried; retrying those just wastes the retry budget on something that
# will never succeed.
_RETRYABLE_STATUS_CODES = {-1, 500, 502, 503, 504}
# Bumped from 3 to 5 after a live run exhausted all 3 attempts on a malformed
# tool-call JSON (gpt-oss:20b, a complex nested new_sources payload) and
# crashed the whole background run - see subagents/sourcing.py's prompt fix
# (batch-per-call -> one-entry-per-call) for the complementary fix that
# reduces how often this needs to fire at all.
_MAX_ATTEMPTS = 5
_BACKOFF_SECONDS = 2


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, ResponseError) and exc.status_code in _RETRYABLE_STATUS_CODES:
        return True
    # langchain_ollama raises a plain ValueError (NOT ollama.ResponseError) when
    # the HTTP stream from the Ollama server yields zero chunks at all - observed
    # live against the office GPU (chat_models.py's _achat_stream_with_aggregation:
    # `raise ValueError("No data received from Ollama stream.")`). This previously
    # wasn't classified as retryable at all, so it skipped the retry loop entirely
    # and crashed the whole graph run on the very first occurrence - almost
    # certainly a transient connection drop to the remote GPU, not a real
    # config/model problem, so it belongs in the same retryable bucket as the
    # ResponseError cases above.
    if isinstance(exc, ValueError) and "No data received from Ollama stream" in str(exc):
        return True
    return False


# When every retry attempt is exhausted on a genuinely retryable error (a
# malformed tool-call JSON from this model, not a config problem), the old
# behavior was to re-raise the original ResponseError. That exception
# originates deep inside a model node with no tool-error handling around it
# at all - and even once it bubbles all the way up through a nested
# subagent's atask() call to the ORCHESTRATOR's own ToolNode, LangGraph's
# default handle_tool_errors (_default_handle_tool_errors in langgraph's
# tool_node.py) only special-cases ToolInvocationError and re-raises
# everything else, including a plain ToolException - confirmed by reading
# that function directly, not assumed. So there is no "catch this as a tool
# error" fix available here; a raised exception at this layer always crashes
# the whole graph run regardless of exception type. The only real fix is to
# never let it escape this wrapper in the first place - return a normal (if
# apologetic) ChatResult/AIMessage instead, so the calling agent loop treats
# this exactly like a model that answered with plain text and no tool call,
# and the run stays alive. This is deliberately ONLY done for exhausted
# RETRYABLE errors (a persistent 4xx - bad model/auth - still raises
# immediately, unretried, same as before, since that's a real configuration
# problem that should surface loudly rather than be silently laundered into
# a confusing subagent response repeated every call).
def _fallback_message(exc: Exception) -> str:
    # Phrased to read sensibly in EITHER of the two places it can surface:
    # relayed up through a nested subagent's atask() as its "final answer" to
    # the orchestrator, or - as actually observed live - as the top-level
    # orchestrator's own model call failing, in which case this text becomes
    # the literal reply shown to the human user in the chat UI. Written in
    # first person / plain language for that reason, not as an internal log
    # line - an earlier version read like a raw diagnostic dump when it
    # happened to be what a real user saw as "the assistant's answer".
    return (
        "I hit a transient error generating my last step - the model produced "
        f"malformed tool-call output after {_MAX_ATTEMPTS} attempts, not a real "
        "data or tool problem. Nothing was changed on disk by this. Please ask "
        f"me to continue or repeat the last request and I'll retry. (Detail: {exc})"
    )


class _RetryingChatOllama(ChatOllama):
    """ChatOllama that retries transient errors (see _RETRYABLE_STATUS_CODES)
    on all four entry points BaseChatModel funnels invoke/ainvoke/stream/
    astream through - used for BOTH local and cloud models.

    The streaming variants only retry if nothing has been yielded yet in the
    failing attempt - once a caller has received a partial chunk there's no
    safe way to "undo" it, so a mid-stream failure past the first chunk still
    raises rather than risking duplicated/garbled output.
    """

    def _generate(self, *args, **kwargs):
        exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                return super()._generate(*args, **kwargs)
            except Exception as e:
                if not _is_retryable(e):
                    raise
                exc = e
                if attempt == _MAX_ATTEMPTS - 1:
                    break
                time.sleep(_BACKOFF_SECONDS * (attempt + 1))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=_fallback_message(exc)))])

    def _stream(self, *args, **kwargs):
        exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            yielded_any = False
            try:
                for chunk in super()._stream(*args, **kwargs):
                    yielded_any = True
                    yield chunk
                return
            except Exception as e:
                if yielded_any or not _is_retryable(e):
                    raise
                exc = e
                if attempt == _MAX_ATTEMPTS - 1:
                    break
                time.sleep(_BACKOFF_SECONDS * (attempt + 1))
        yield ChatGenerationChunk(message=AIMessageChunk(content=_fallback_message(exc)))

    async def _agenerate(self, *args, **kwargs):
        exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                return await super()._agenerate(*args, **kwargs)
            except Exception as e:
                if not _is_retryable(e):
                    raise
                exc = e
                if attempt == _MAX_ATTEMPTS - 1:
                    break
                await asyncio.sleep(_BACKOFF_SECONDS * (attempt + 1))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=_fallback_message(exc)))])

    async def _astream(self, *args, **kwargs):
        exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            yielded_any = False
            try:
                async for chunk in super()._astream(*args, **kwargs):
                    yielded_any = True
                    yield chunk
                return
            except Exception as e:
                if yielded_any or not _is_retryable(e):
                    raise
                exc = e
                if attempt == _MAX_ATTEMPTS - 1:
                    break
                await asyncio.sleep(_BACKOFF_SECONDS * (attempt + 1))
        yield ChatGenerationChunk(message=AIMessageChunk(content=_fallback_message(exc)))


class _ThrottledCloudChatOllama(_RetryingChatOllama):
    """_RetryingChatOllama plus a process-wide lock serializing every call -
    cloud-only, since a free-tier key allows just one in-flight call at a
    time. Sync and async calls are only serialized against calls of the same
    kind (separate locks) - acceptable here since real usage is either the
    fully-async langgraph server or a single standalone test script, never
    both against the same key at once.
    """

    def _generate(self, *args, **kwargs):
        with _CLOUD_CALL_LOCK:
            return super()._generate(*args, **kwargs)

    def _stream(self, *args, **kwargs):
        with _CLOUD_CALL_LOCK:
            yield from super()._stream(*args, **kwargs)

    async def _agenerate(self, *args, **kwargs):
        async with _CLOUD_CALL_ASYNC_LOCK:
            return await super()._agenerate(*args, **kwargs)

    async def _astream(self, *args, **kwargs):
        async with _CLOUD_CALL_ASYNC_LOCK:
            async for chunk in super()._astream(*args, **kwargs):
                yield chunk


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
        # disable_streaming="tool_calling" (a BaseChatModel built-in, not
        # Ollama-specific) forces _generate/_agenerate instead of _stream/
        # _astream whenever tools are bound - i.e. on essentially every call
        # in this codebase, since every subagent has tools. Observed live on
        # a local model: streamed tool-call JSON gets reassembled chunk-by-
        # chunk, and a small/quantized model's output occasionally doesn't
        # reassemble into valid JSON (truncated mid-argument, or a stray
        # trailing character) even though the retry logic above already
        # covers that failure mode - this avoids the fragile reassembly path
        # entirely rather than just retrying around it.
        try:
            if api_key:
                return _ThrottledCloudChatOllama(
                    model=model_name,
                    base_url=base_url,
                    client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}},
                    disable_streaming="tool_calling",
                )
            return _RetryingChatOllama(
                model=model_name, base_url=base_url, disable_streaming="tool_calling"
            )
        except Exception as exc:  # noqa: BLE001 - degrade, don't crash startup
            print(f"[model_builder] failed to init {model_spec!r} ({exc}); falling back to caller's default.")
            return None

    try:
        return init_chat_model(model_spec)
    except Exception as exc:  # noqa: BLE001 - degrade, don't crash startup
        print(f"[model_builder] failed to init {model_spec!r} ({exc}); falling back to caller's default.")
        return None
