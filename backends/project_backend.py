"""Shared virtual filesystem for the orchestrator and every subagent.

Everything under /workspace/ is routed to real disk (RUN_ARTIFACTS_DIR) so a
judge/user can open plan.md, sources.json, dataset/, runs/, eval_report.md
etc. directly after a run - see §7 of the architecture doc. Anything written
outside /workspace/ falls back to ephemeral in-state storage.
"""

import os
from pathlib import Path

from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

RUN_ARTIFACTS_DIR = os.environ.get(
    "RUN_ARTIFACTS_DIR", str(Path(__file__).resolve().parent.parent / "run_artifacts")
)

Path(RUN_ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)

project_backend = CompositeBackend(
    default=StateBackend(),
    routes={
        # virtual_mode=True is required here, not optional: CompositeBackend
        # strips the "/workspace/" prefix and forwards normalized paths to
        # this backend, and deepagents' own docs say virtual_mode is exactly
        # for that case. Without it (the default), absolute paths bypass
        # root_dir entirely - a tool call with path="/" resolved to the real
        # filesystem root and crashed a run walking C:\$Recycle.Bin. With
        # virtual_mode=True, every path is anchored under root_dir and a
        # resolved path escaping it raises ValueError instead of touching
        # real files outside run_artifacts/.
        "/workspace/": FilesystemBackend(root_dir=RUN_ARTIFACTS_DIR, virtual_mode=True),
    },
)
