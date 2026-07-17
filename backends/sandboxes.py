"""Sandbox backends for subagents that need real (GPU) compute via execute().

Two separate sandboxes, per §8 of the architecture doc:
- annotation: short-lived, CPU/small-GPU, just runs a zero-shot detector batch.
- training: longer-lived, GPU required, reused by eval-agent afterward so the
  trained weights are already local to that sandbox.

deepagents ships no built-in cloud sandbox (only LocalShellBackend and
LangSmithSandbox) - `ModalSandbox` below is a real BaseSandbox subclass
wrapping the `modal` Python SDK, following the same pattern as deepagents'
own nvidia_deep_agent example (Modal for GPU-accelerated subagent execution).

Each ModalSandbox instance lazily creates ONE long-lived Modal Sandbox
container on first use and reuses it for every execute()/upload_files()/
download_files() call - it is not spun up per call. The container is
terminated at process exit (best-effort; Modal will also reap it once its
own `timeout` elapses if the process dies uncleanly).
"""

from __future__ import annotations

import atexit
import os
from typing import Final

import modal
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

_APP_NAME: Final = os.environ.get("MODAL_APP_NAME", "yolo-deep-agent")


class ModalSandbox(BaseSandbox):
    """A Modal Sandbox-backed execute()/upload_files()/download_files()."""

    def __init__(self, *, image: str, gpu: str | None = None, timeout: int = 600) -> None:
        self._image_ref = image
        self._gpu = gpu
        self._timeout = timeout
        self._sandbox: modal.Sandbox | None = None

    def _ensure_sandbox(self) -> modal.Sandbox:
        if self._sandbox is None:
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


training_sandbox_backend = ModalSandbox(
    image=os.environ.get("TRAINING_SANDBOX_IMAGE", "ultralytics/ultralytics:latest-python"),
    gpu=os.environ.get("TRAINING_SANDBOX_GPU", "A10G"),
    timeout=int(os.environ.get("TRAINING_SANDBOX_TIMEOUT", str(60 * 45))),
)

annotation_sandbox_backend = ModalSandbox(
    image=os.environ.get("ANNOTATION_SANDBOX_IMAGE", "python:3.11-slim"),
    gpu=os.environ.get("ANNOTATION_SANDBOX_GPU", "T4"),
    timeout=int(os.environ.get("ANNOTATION_SANDBOX_TIMEOUT", str(60 * 15))),
)
