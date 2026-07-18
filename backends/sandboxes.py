"""Sandbox backends for subagents that need real compute via execute().

Two separate sandboxes, per §8 of the architecture doc:
- annotation: short-lived, CPU/small-GPU, just runs a zero-shot detector batch.
- training: longer-lived, reused by eval-agent afterward so the trained
  weights are already local to that sandbox.

Two backend choices, selected via TRAINING_BACKEND ("local" default, "modal"):

- **local** (default) - `LocalShellBackend` (a real deepagents backend, not
  custom code), rooted at the SAME RUN_ARTIFACTS_DIR real disk directory
  project_backend uses, with virtual_mode=True mirroring project_backend's own
  convention (backends/project_backend.py). Since it's the same directory,
  files training-agent/eval-agent need (dataset/, model_choice.json) are
  already there with no upload step, and whatever they write back
  (runs/train/status.md, eval_report.md) is immediately visible to the
  orchestrator/UI with no download step either -
  subagents/training.py and subagents/eval.py pass empty upload_paths/
  download_paths for this reason. No cost, no API key, no GPU - runs
  directly on this machine via `subprocess.run(shell=True)`. Needs
  `ultralytics`/`supervision` installed wherever that subprocess's PATH
  resolves `python`/`pip` to (this project's own .venv, since inherit_env=True
  passes through the real environment - see pyproject.toml's dependencies).

  **Security note (from LocalShellBackend's own docstring): this runs
  UNRESTRICTED shell commands with your real user permissions, no sandboxing
  at all** - training-agent/eval-agent's inner agents (subagents/
  sandbox_subagent.py) set `interrupt_on={"execute": ...}` on themselves so
  every shell command surfaces for your approval before it actually runs,
  since deepagents' own docs "STRONGLY RECOMMEND" HITL as the safeguard for
  this backend.

- **modal** - `ModalSandbox` below, a real BaseSandbox subclass wrapping the
  `modal` Python SDK for actual isolated GPU compute (real cost, needs
  MODAL_TOKEN_ID/MODAL_TOKEN_SECRET). Each instance lazily creates ONE
  long-lived Modal Sandbox container on first execute()/etc. call and reuses
  it for the rest of the process's life. Switch back via
  `TRAINING_BACKEND=modal` once real GPU training is actually wanted.
"""

from __future__ import annotations

import atexit
import os
from typing import Final

from deepagents.backends import LocalShellBackend
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

from backends.project_backend import RUN_ARTIFACTS_DIR

_APP_NAME: Final = os.environ.get("MODAL_APP_NAME", "yolo-deep-agent")


class ModalSandbox(BaseSandbox):
    """A Modal Sandbox-backed execute()/upload_files()/download_files()."""

    def __init__(self, *, image: str, gpu: str | None = None, timeout: int = 600) -> None:
        self._image_ref = image
        self._gpu = gpu
        self._timeout = timeout
        self._sandbox = None

    def _ensure_sandbox(self):
        if self._sandbox is None:
            import modal  # noqa: PLC0415 - only import the (optional-ish) modal SDK if actually used

            app = modal.App.lookup(_APP_NAME, create_if_missing=True)
            image = modal.Image.from_registry(self._image_ref, add_python="3.11")
            self._sandbox = modal.Sandbox.create(
                "sleep",
                "infinity",
                app=app,
                image=image,
                gpu=self._gpu,
                timeout=self._timeout,
            )
            atexit.register(self._terminate)
        return self._sandbox

    def _terminate(self) -> None:
        if self._sandbox is not None:
            try:
                self._sandbox.terminate()
            except Exception:  # best-effort cleanup at interpreter exit
                pass

    @property
    def id(self) -> str:
        return self._ensure_sandbox().object_id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        sandbox = self._ensure_sandbox()
        process = sandbox.exec("bash", "-c", command, timeout=timeout)
        output = process.stdout.read() + process.stderr.read()
        process.wait()
        return ExecuteResponse(output=output, exit_code=process.returncode)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        fs = self._ensure_sandbox().filesystem
        responses: list[FileUploadResponse] = []
        for path, content in files:
            try:
                parent = os.path.dirname(path)
                if parent:
                    fs.make_directory(parent, create_parents=True)
                fs.write_bytes(content, path)
                responses.append(FileUploadResponse(path=path))
            except Exception as exc:
                responses.append(FileUploadResponse(path=path, error=str(exc)))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        fs = self._ensure_sandbox().filesystem
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                content = fs.read_bytes(path)
                responses.append(FileDownloadResponse(path=path, content=content))
            except FileNotFoundError:
                responses.append(FileDownloadResponse(path=path, error="file_not_found"))
            except Exception as exc:
                responses.append(FileDownloadResponse(path=path, error=str(exc)))
        return responses


def _build_training_backend():
    if os.environ.get("TRAINING_BACKEND", "local") == "modal":
        return ModalSandbox(
            image=os.environ.get("TRAINING_SANDBOX_IMAGE", "ultralytics/ultralytics:latest-python"),
            gpu=os.environ.get("TRAINING_SANDBOX_GPU", "A10G"),
            timeout=int(os.environ.get("TRAINING_SANDBOX_TIMEOUT", str(60 * 45))),
        )
    return LocalShellBackend(
        root_dir=RUN_ARTIFACTS_DIR,
        virtual_mode=True,
        inherit_env=True,
        timeout=int(os.environ.get("TRAINING_SANDBOX_TIMEOUT", str(60 * 20))),
    )


training_sandbox_backend = _build_training_backend()

annotation_sandbox_backend = LocalShellBackend(
    root_dir=RUN_ARTIFACTS_DIR,
    virtual_mode=True,
    inherit_env=True,
    timeout=int(os.environ.get("ANNOTATION_SANDBOX_TIMEOUT", str(60 * 15))),
)
