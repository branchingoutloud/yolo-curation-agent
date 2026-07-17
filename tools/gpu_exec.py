"""Run a YOLO driver script inside the ultralytics GPU Docker container.

This is the single place local-GPU-server execution is wired. The orchestrator
process NEVER imports torch/ultralytics (per CLAUDE.md - torch is intentionally
kept out of the orchestrator venv); it only shells out to `docker run`, and all
the heavy ML work happens inside the container against the local GPU.

Because the agent runs on the *same machine* as the GPU, files cross in/out
purely through a bind mount of RUN_ARTIFACTS_DIR at /workspace - there is no
upload/download step and no cross-filesystem gap. This is the whole reason
running `langgraph dev` on the GPU server (not over SSH) is simpler and more
robust than any remote/hosted sandbox: the dataset the orchestrator wrote to
run_artifacts/ is already sitting next to the GPU.

Requirements on the host running the agent:
- Docker + the NVIDIA Container Toolkit (so `--gpus` works).
- The image pulled once: `docker pull ultralytics/ultralytics:latest`.

Driver scripts live in tools/gpu_drivers/ and are mounted read-only at /drivers;
they are executed *inside* the container (where torch/ultralytics exist), never
imported by this process.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from backends.project_backend import RUN_ARTIFACTS_DIR
from tools.docker_sandbox import CONTAINER_NAME, DockerSandboxManager

_DRIVERS_DIR = Path(__file__).resolve().parent / "gpu_drivers"

# All overridable via .env (see .env.example). Defaults suit a single-GPU box.
DEFAULT_IMAGE = os.environ.get("YOLO_TRAIN_IMAGE", "ultralytics/ultralytics:latest")
DEFAULT_GPUS = os.environ.get("YOLO_TRAIN_GPUS", "all")
DEFAULT_SHM = os.environ.get("YOLO_TRAIN_SHM", "8g")

_OUTPUT_TAIL_CHARS = 4000


class GpuExecError(Exception):
    """Raised when the container run fails, times out, or docker is missing.

    The message carries a tail of the captured output so the calling @tool can
    return it verbatim as plain text (an uncaught exception inside a @tool
    crashes the whole LangGraph run - see tools/dataset_builder.py).
    """


def _tail(text: str) -> str:
    return text[-_OUTPUT_TAIL_CHARS:] if text else ""


def run_driver(script_name: str, args: list[str], *, timeout: int) -> str:
    """Run tools/gpu_drivers/<script_name> in the GPU container; return its output.

    Args:
        script_name: file name under tools/gpu_drivers/ (e.g. "train_driver.py").
        args: positional string args passed to the driver after its path.
        timeout: hard wall-clock limit in seconds for the whole container run.

    Returns:
        Combined stdout+stderr on success (exit code 0).

    Raises:
        GpuExecError: on non-zero exit, timeout, or docker-not-found.
    """
    # 1. Ensure the persistent sandbox container is running
    DockerSandboxManager.ensure_started()

    # 2. Execute the driver script inside the long-lived container via docker exec
    cmd: list[str] = [
        "docker",
        "exec",
        CONTAINER_NAME,
        "python",
        f"/drivers/{script_name}",
        *args,
    ]

    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, args are ints/enums
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        msg = (
            "`docker` was not found on PATH. training/eval run inside the "
            "ultralytics Docker image - install Docker + the NVIDIA Container "
            "Toolkit and run the agent on the GPU server itself."
        )
        raise GpuExecError(msg) from exc
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "")
        msg = f"container run timed out after {timeout}s. Output tail:\n{_tail(out)}"
        raise GpuExecError(msg) from exc

    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        msg = f"container exited with code {proc.returncode}. Output tail:\n{_tail(output)}"
        raise GpuExecError(msg)
    return output
