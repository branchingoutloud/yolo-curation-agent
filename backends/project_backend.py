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
        "/workspace/": FilesystemBackend(root_dir=RUN_ARTIFACTS_DIR, virtual_mode=True),
    },
)
