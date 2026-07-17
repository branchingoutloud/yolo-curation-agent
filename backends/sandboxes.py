"""Sandbox backend selection for subagents that need real execute().

The architecture doc (§8) describes per-subagent Modal GPU sandboxes
(`ModalSandbox`), but that class does not exist in the installed deepagents
version (0.6.12 only ships the abstract `BaseSandbox`) - and this version of
`SubAgent` also has no per-subagent `backend` override at all (a `"backend"`
key on a subagent dict is silently ignored; see subagents/training.py). So
there is exactly one sandbox for the whole graph: whatever is passed as
`backend=` to `create_deep_agent` in agent.py (see backends/project_backend.py).

Two options are wired here, selected via SANDBOX_BACKEND:

- "local" (default) - `LocalShellBackend`, runs execute() directly on this
  host. Zero setup, works immediately with the already-installed
  `ultralytics`/`torch`. No isolation - per deepagents' own docs this is only
  appropriate for local dev, which is exactly today's use case (testing
  against Ollama Cloud models on a personal machine). Not for judged/shared
  or production runs.
- "docker" - `DockerSandbox` (backends/docker_sandbox.py), execute() runs
  inside the container started by `docker compose -f docker/docker-compose.yml
  up -d`. Real process isolation; requires Docker Desktop running.
"""

import os
from pathlib import Path

from deepagents.backends import LocalShellBackend

RUN_ARTIFACTS_DIR = os.environ.get(
    "RUN_ARTIFACTS_DIR", str(Path(__file__).resolve().parent.parent / "run_artifacts")
)
Path(RUN_ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)

SANDBOX_BACKEND = os.environ.get("SANDBOX_BACKEND", "local").strip().lower()
SANDBOX_TIMEOUT = int(os.environ.get("SANDBOX_TIMEOUT", str(60 * 30)))


def build_sandbox_backend():
    """Return the single sandbox-capable backend used for every subagent's execute()."""
    if SANDBOX_BACKEND == "docker":
        from backends.docker_sandbox import DockerSandbox

        return DockerSandbox(
            container_name=os.environ.get("TRAINING_CONTAINER_NAME", "yolo-training-sandbox"),
            root_dir=RUN_ARTIFACTS_DIR,
            workdir="/workspace",
            timeout=SANDBOX_TIMEOUT,
        )

    if SANDBOX_BACKEND != "local":
        print(f"[sandboxes] Unknown SANDBOX_BACKEND={SANDBOX_BACKEND!r}, falling back to 'local'.")

    return LocalShellBackend(
        root_dir=RUN_ARTIFACTS_DIR,
        virtual_mode=True,
        inherit_env=True,  # need PATH (python/yolo CLI), OLLAMA_*, etc.
        timeout=SANDBOX_TIMEOUT,
    )


sandbox_backend = build_sandbox_backend()
