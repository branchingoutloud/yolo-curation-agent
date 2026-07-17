"""Model construction for the orchestrator (and, by inheritance, every subagent).

A bare `"provider:model"` string works fine for providers `init_chat_model`
already knows how to authenticate from standard env vars (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, ...). It does NOT work for Ollama Cloud: that needs a custom
`base_url` plus an `Authorization: Bearer <key>` header, which
`init_chat_model("ollama:...")` has no way to supply. Passing the bare string
through to `create_deep_agent` silently resolves to a `ChatOllama` pointed at
the local default (`http://localhost:11434`, no auth) - it fails against
Ollama Cloud, not because of a bad key, but because it's talking to the wrong
place.

`build_default_model()` constructs the real, fully-configured model object
once. Passing that *object* (not a string) as `create_deep_agent(model=...)`
matters beyond the top-level call too: deepagents resolves each subagent's
model as `spec.get("model", model)` — if a subagent doesn't set its own
`model`, it inherits whatever `model` is here. Passing an already-constructed
`BaseChatModel` means every subagent reuses that exact same authenticated
instance; passing a string would make deepagents re-run `init_chat_model` per
subagent and drop the custom headers all over again.
"""

import os
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


@lru_cache(maxsize=1)
def build_default_model() -> "BaseChatModel":
    """Build the model every agent/subagent uses unless it names its own.

    Reads `ORCHESTRATOR_MODEL` (default: `"ollama:gpt-oss:20b"`):
      - `"ollama:<model>"` — `ChatOllama` against `OLLAMA_BASE_URL` with an
        `Authorization: Bearer <OLLAMA_API_KEY>` header (Ollama Cloud).
        Requires both `OLLAMA_BASE_URL` and `OLLAMA_API_KEY` to be set.
      - anything else (e.g. `"anthropic:claude-sonnet-5"`) — passed straight
        to `init_chat_model`; those providers authenticate from their own
        standard env vars, so a bare string is fine.
    """
    spec = os.environ.get("ORCHESTRATOR_MODEL", "ollama:gpt-oss:20b")
    provider, sep, model_name = spec.partition(":")

    if sep and provider == "ollama":
        return _build_ollama_model(model_name)

    from langchain.chat_models import init_chat_model

    return init_chat_model(spec)


def _build_ollama_model(model_name: str) -> "BaseChatModel":
    from langchain_ollama import ChatOllama

    base_url = os.environ.get("OLLAMA_BASE_URL")
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not base_url or not api_key:
        msg = (
            "ORCHESTRATOR_MODEL uses the 'ollama:' provider but OLLAMA_BASE_URL "
            "and/or OLLAMA_API_KEY are not set — see .env."
        )
        raise RuntimeError(msg)

    return ChatOllama(
        model=model_name,
        base_url=base_url,
        client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}},
    )
