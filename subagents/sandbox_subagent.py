"""Bridges a sandbox-backed subagent (training-agent, eval-agent) into the
orchestrator's `task()` delegation despite deepagents==0.6.12 having no
per-subagent `"backend"` override (see tools/dataset_builder.py's module
docstring / CLAUDE.md's Known Stubs for that limitation in full).

The trick: `CompiledSubAgent` (a `deepagents.SubAgent` alternative) accepts
any pre-built `Runnable` as `"runnable"`, and per graph.py's subagent-building
loop, `CompiledSubAgent` entries are used AS-IS - never rebuilt against the
orchestrator's shared `backend=`. So instead of a declarative `SubAgent`
dict (which would get silently forced onto `project_backend` like every
other subagent), training-agent/eval-agent are built as their OWN, separate
`create_deep_agent(backend=sandbox_backend, ...)` graphs - which DO get a
real `execute` tool, since `FilesystemMiddleware` adds one whenever the
backend passed to *that specific* `create_deep_agent` call implements
`SandboxBackendProtocol` (true for `ModalSandbox`, and would be equally true
for `LangSmithSandbox` - this wrapper is backend-agnostic, swap the backend
instance to switch providers with no other code change).

The catch with giving a subagent its own separate backend: its filesystem
tools (ls/read_file/write_file/edit_file, auto-added by FilesystemMiddleware)
now operate against the SANDBOX's filesystem, not project_backend's real
disk under RUN_ARTIFACTS_DIR - so whatever the orchestrator/dataset-agent
already wrote there (dataset/data.yaml, model_choice.json) is invisible to
it unless copied in first, and whatever it writes back (runs/train/
status.md, metrics.json) is invisible to the orchestrator/eval-agent
afterward unless copied out. `build_sandbox_subagent()` does exactly that
copy-in/copy-out via the sandbox's own upload_files()/download_files()
(already implemented in backends/sandboxes.py), bracketing one inner-agent
invocation - using the same `/workspace/...` absolute-path convention on
both sides so the same path string means the same thing in a prompt whether
it resolves to real disk (project_backend) or the sandbox's container
filesystem.

Not yet exercised against a real Modal run (spinning up a GPU sandbox costs
real money/time - deliberately not triggered without an explicit go-ahead).
The wiring itself (execute tool present, upload/download bridge structure)
is verified by inspection/construction, not a live end-to-end training run.
"""

from __future__ import annotations

import asyncio
from typing import Any

from deepagents import create_deep_agent
from deepagents.middleware.subagents import CompiledSubAgent
from langchain_core.runnables import RunnableLambda

from tools.workspace_paths import workspace_path


def _collect_upload_files(virtual_path: str) -> list[tuple[str, bytes]]:
    """Read a /workspace/... file or directory tree off real disk, keyed by
    the SAME /workspace/... absolute path it should land at inside the
    sandbox. Silently yields nothing for a path that doesn't exist yet
    (e.g. model_choice.json before gate 2) - the inner agent's own prompt is
    responsible for reporting a missing file as an error, not this bridge.
    """
    local_root = workspace_path(virtual_path)
    virtual_root = virtual_path.rstrip("/")
    if local_root.is_file():
        return [(virtual_root, local_root.read_bytes())]
    if not local_root.is_dir():
        return []

    files: list[tuple[str, bytes]] = []
    for local_file in local_root.rglob("*"):
        if local_file.is_file():
            rel = local_file.relative_to(local_root).as_posix()
            files.append((f"{virtual_root}/{rel}", local_file.read_bytes()))
    return files


def _write_download_result(virtual_path: str, content: bytes) -> None:
    local_path = workspace_path(virtual_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(content)


def build_sandbox_subagent(
    *,
    name: str,
    description: str,
    system_prompt: str,
    sandbox_backend: Any,
    model: Any = None,
    upload_paths: list[str] = (),
    download_paths: list[str] = (),
) -> CompiledSubAgent:
    """Build a CompiledSubAgent whose inner agent has a REAL execute() tool
    bound to `sandbox_backend`, with real-disk files bridged in/out of it.

    upload_paths/download_paths are /workspace/-prefixed virtual paths
    (files or whole directories), same convention as every other subagent
    in this repo. Uploaded before the inner agent runs; downloaded after it
    finishes - including on failure, so partial progress (e.g. a status.md
    written before a later step errored) isn't lost.
    """
    inner_kwargs: dict[str, Any] = {
        "system_prompt": system_prompt,
        "tools": [],
        "backend": sandbox_backend,
    }
    if model is not None:
        inner_kwargs["model"] = model
    inner_agent = create_deep_agent(**inner_kwargs)

    def _upload_sync() -> None:
        upload_batch: list[tuple[str, bytes]] = []
        for virtual_path in upload_paths:
            upload_batch.extend(_collect_upload_files(virtual_path))
        if upload_batch:
            sandbox_backend.upload_files(upload_batch)

    def _download_sync() -> None:
        for response in sandbox_backend.download_files(list(download_paths)):
            if response.content is not None:
                _write_download_result(response.path, response.content)
            # A missing/failed download (e.g. training never got far enough
            # to write metrics.json) is expected on a partial run - the
            # inner agent's own final message is what reports that, not
            # this bridge.

    async def _run(state: dict, config: dict | None = None) -> dict:
        # Both the local file I/O (workspace_path().resolve(), read_bytes())
        # and the sandbox_backend calls (modal's SDK is synchronous) are
        # blocking - langgraph dev's blockbuster middleware correctly flags
        # a bare synchronous call inside an async function as an event-loop
        # stall, so every blocking step here is offloaded via
        # asyncio.to_thread rather than called directly.
        if upload_paths:
            await asyncio.to_thread(_upload_sync)

        # Without this, the inner agent inherits whatever recursion_limit the
        # PARENT graph invocation was given (LangGraph's default is 100) -
        # observed live: a real training-agent run hit that ceiling with no
        # runs/ directory or status.md ever written, i.e. it was stuck
        # cycling through checks/retries, not making 100 steps of genuine
        # progress. Training/eval legitimately need more turns than a typical
        # planning/sourcing delegation (environment checks, a long-running
        # execute() call, status polling), so this raises the ceiling - but
        # see the tightened system_prompt in subagents/training.py, which is
        # the more important fix for the actual looping behavior itself.
        inner_config = {**(config or {}), "recursion_limit": max((config or {}).get("recursion_limit", 0), 150)}

        try:
            return await inner_agent.ainvoke(state, config=inner_config)
        finally:
            if download_paths:
                await asyncio.to_thread(_download_sync)

    return {
        "name": name,
        "description": description,
        "runnable": RunnableLambda(_run),
    }
