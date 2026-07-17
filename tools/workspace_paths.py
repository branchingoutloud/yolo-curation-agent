"""Shared `/workspace/...` virtual-path resolution for custom tools that read
or write real files directly (bypassing the deepagents filesystem-tool
abstraction, which is impractical for bulk file/directory operations).

Used by dataset_builder.py, sourcing_builder.py, and planning_builder.py -
factored out here once all three needed the identical logic, rather than
duplicating it per module.
"""

from pathlib import Path

from backends.project_backend import RUN_ARTIFACTS_DIR


def workspace_path(virtual_path: str) -> Path:
    """Resolve a `/workspace/...` virtual path to its real path on disk.

    Mirrors the same prefix-stripping + root-anchoring that
    `CompositeBackend`/`FilesystemBackend(virtual_mode=True)` apply for the
    `/workspace/` route in `backends/project_backend.py`.
    """
    normalized = virtual_path.strip()
    if not normalized.startswith("/workspace/") and normalized != "/workspace":
        msg = f"Path must be under /workspace/ (got {virtual_path!r}) - that's the only route persisted to real disk."
        raise ValueError(msg)
    relative = normalized.removeprefix("/workspace/").removeprefix("/workspace")
    root = Path(RUN_ARTIFACTS_DIR).resolve()
    resolved = (root / relative).resolve()
    if root not in resolved.parents and resolved != root:
        msg = f"Resolved path {resolved} escapes the workspace root {root}"
        raise ValueError(msg)
    return resolved
