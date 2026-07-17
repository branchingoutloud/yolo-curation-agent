"""Shared virtual filesystem + sandbox for the orchestrator and every subagent.

Everything under /workspace/ is routed to real disk (RUN_ARTIFACTS_DIR) so a
judge/user can open plan.md, sources.json, dataset/, runs/, eval_report.md
etc. directly after a run - see §7 of the architecture doc.

Important correction vs. the architecture doc: `CompositeBackend.execute()`
(installed deepagents==0.6.12) *always* delegates to `default`, never to a
routed backend - so `default` must itself be the sandbox-capable backend, not
an ephemeral `StateBackend`, or no subagent (including training-agent) gets a
working `execute()` tool at all. `default` is set to the same sandbox backend
that's routed for /workspace/ (see backends/sandboxes.py), so:
  - reads/writes under /workspace/ behave exactly as before (prefix stripped,
    persisted to RUN_ARTIFACTS_DIR)
  - execute() now actually runs (on the host, or in Docker - see
    SANDBOX_BACKEND in backends/sandboxes.py) instead of raising
    NotImplementedError
  - paths outside /workspace/ also land under RUN_ARTIFACTS_DIR now (no more
    silent StateBackend scratch space) - harmless for this project, since
    every subagent is instructed to write under /workspace/ anyway.

Second correction: `SkillsMiddleware` sources are *virtual backend paths*
(POSIX-style, e.g. "/skills/project/"), not raw host filesystem paths -
"Sources point to skill directories in the backend" (see
deepagents.middleware.skills' module docstring). agent.py previously passed
an absolute host path (`str(Path(__file__).parent / "skills")`) straight
through as `skills=[...]`; with the old `default=StateBackend()` that
silently matched nothing (0 skills loaded, no error - `StateBackend.ls()`
just treats an unfamiliar key as an empty directory). With the new
sandbox-backed default, the same call raises `ValueError: ... outside root
directory`, since a real FilesystemBackend actually enforces `root_dir`.
Fix: route the *virtual* path "/skills/" to a FilesystemBackend rooted at
this repo's real skills/ directory, and pass "/skills" (not an absolute
host path) to `skills=[...]` in agent.py / the test scripts.
"""

from pathlib import Path

from deepagents.backends import CompositeBackend, FilesystemBackend

from backends.sandboxes import RUN_ARTIFACTS_DIR, sandbox_backend

SKILLS_DIR = str(Path(__file__).resolve().parent.parent / "skills")

project_backend = CompositeBackend(
    default=sandbox_backend,
    routes={
        "/workspace/": sandbox_backend,
        "/skills/": FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True),
    },
)

__all__ = ["RUN_ARTIFACTS_DIR", "project_backend"]
