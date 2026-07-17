"""Preflight check for Ollama (cloud or local) before running a full
subagent test against it - confirms the server/API is reachable AND that
the configured model actually responds, before spending several agent turns
just to discover a connection/auth error or a 404 on the first tool call.

Checks whichever of ORCHESTRATOR_MODEL / DATASET_AGENT_MODEL is set to
"ollama:...". If OLLAMA_API_KEY is set, authenticates as Ollama Cloud
(Bearer header); otherwise assumes a local/unauthenticated server.

Usage:
    python scripts/check_ollama.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import requests

base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
api_key = os.environ.get("OLLAMA_API_KEY")

model_spec = os.environ.get("ORCHESTRATOR_MODEL", "")
if not model_spec.startswith("ollama:"):
    model_spec = os.environ.get("DATASET_AGENT_MODEL", "")
wanted_model = model_spec.removeprefix("ollama:") if model_spec.startswith("ollama:") else None

headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

print(f"OLLAMA_BASE_URL = {base_url}")
print(f"OLLAMA_API_KEY  = {'set (cloud mode)' if api_key else '(unset - local mode)'}")
print(f"model under test = {model_spec or '(unset)'}")

if not wanted_model:
    print(
        "\nNeither ORCHESTRATOR_MODEL nor DATASET_AGENT_MODEL starts with 'ollama:' - "
        "nothing to check. Set one to e.g. 'ollama:gpt-oss:120b' first."
    )
    sys.exit(1)

# /api/tags lists locally-pulled models - unverified whether Ollama Cloud
# exposes the same endpoint for its hosted catalog, so this is best-effort.
# A failure here isn't fatal; the /api/chat call below is the real check,
# since that's what every subagent actually does regardless.
try:
    response = requests.get(f"{base_url}/api/tags", headers=headers, timeout=10)
    response.raise_for_status()
    models = [m["name"] for m in response.json().get("models", [])]
    print(f"\n/api/tags reachable. {len(models)} model(s) listed:")
    for m in models:
        print(f"  - {m}")
except requests.RequestException as exc:
    print(f"\n/api/tags check inconclusive ({exc}) - trying a real chat call instead.")

try:
    chat_response = requests.post(
        f"{base_url}/api/chat",
        headers=headers,
        json={
            "model": wanted_model,
            "messages": [{"role": "user", "content": "Reply with just the word: ok"}],
            "stream": False,
        },
        timeout=60,
    )
    chat_response.raise_for_status()
    content = chat_response.json().get("message", {}).get("content", "")
    print(f"\nOK: '{wanted_model}' responded: {content!r}")
except requests.RequestException as exc:
    print(f"\nFAILED to chat with '{wanted_model}' at {base_url}: {exc}")
    response_obj = getattr(exc, "response", None)
    if response_obj is not None:
        print(f"Response body: {response_obj.text[:500]}")
    print(
        "Check: is the model tag right (cloud models may need a '-cloud' suffix - see "
        "tools/model_builder.py), is OLLAMA_API_KEY valid, is base_url right for cloud "
        "(https://ollama.com) vs local (http://localhost:11434 or the office GPU's address)?"
    )
    sys.exit(1)
