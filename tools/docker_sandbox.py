"""Lifecycle manager for the long-lived local Docker sandbox container.

Provides methods to ensure the container is pulled, started, and cleanly terminated.
"""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
from pathlib import Path

from backends.project_backend import RUN_ARTIFACTS_DIR

CONTAINER_NAME = "yolo-training-sandbox"
_DRIVERS_DIR = Path(__file__).resolve().parent / "gpu_drivers"


class DockerSandboxManager:
    """Manages the long-lived background Docker container used for YOLO training/eval."""

    _is_registered = False

    @classmethod
    def ensure_started(cls) -> None:
        """Check if the container is running; if not, pull image and start it."""
        image = os.environ.get("YOLO_TRAIN_IMAGE", "ultralytics/ultralytics:latest")
        gpus = os.environ.get("YOLO_TRAIN_GPUS", "all")
        shm_size = os.environ.get("YOLO_TRAIN_SHM", "8g")
        device = os.environ.get("YOLO_TRAIN_DEVICE", "0")

        # 1. Check if the container is already running
        proc_running = subprocess.run(
            ["docker", "ps", "-q", "-f", f"name=^{CONTAINER_NAME}$"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc_running.stdout.strip():
            # Already running, nothing to do
            cls._register_cleanup()
            return

        # 2. Check if a stopped/stale container exists and remove it
        proc_exists = subprocess.run(
            ["docker", "ps", "-a", "-q", "-f", f"name=^{CONTAINER_NAME}$"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc_exists.stdout.strip():
            print(f"[docker_sandbox] Removing stale container '{CONTAINER_NAME}'...")
            subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True, check=False)

        # 3. Verify Docker image exists locally; if not, pull it
        print(f"[docker_sandbox] Checking if image '{image}' is available locally...")
        proc_inspect = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            check=False,
        )
        if proc_inspect.returncode != 0:
            print(f"[docker_sandbox] Image '{image}' not found locally. Pulling it (this may take a few minutes)...")
            pull_proc = subprocess.run(["docker", "pull", image], check=False)
            if pull_proc.returncode != 0:
                print(f"[docker_sandbox] Warning: Failed to pull '{image}'. Attempting to run anyway...")

        # 4. Start the container in detached mode
        workspace = Path(RUN_ARTIFACTS_DIR).resolve()
        workspace.mkdir(parents=True, exist_ok=True)

        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER_NAME,
            f"--shm-size={shm_size}",
            "-v",
            f"{workspace}:/workspace",
            "-v",
            f"{_DRIVERS_DIR}:/drivers:ro",
            "-w",
            "/workspace",
            "-e",
            "HOME=/workspace",
            "-e",
            "YOLO_CONFIG_DIR=/workspace/.ultralytics",
            "-e",
            "MPLCONFIGDIR=/workspace/.mpl",
            "-e",
            "TORCH_HOME=/workspace/.torch",
        ]

        # Enable GPU options only if we are not targetting CPU
        if gpus and device != "cpu":
            cmd.insert(3, f"--gpus={gpus}")

        if hasattr(os, "getuid"):
            cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]

        # Infinite sleep equivalent to keep container running in background
        cmd += [image, "tail", "-f", "/dev/null"]

        print(f"[docker_sandbox] Starting long-lived sandbox container '{CONTAINER_NAME}'...")
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to start Docker sandbox container: {res.stderr}")

        cls._register_cleanup()
        print(f"[docker_sandbox] Sandbox container '{CONTAINER_NAME}' is running.")

    @classmethod
    def stop(cls) -> None:
        """Stop and remove the running container."""
        print(f"\n[docker_sandbox] Cleaning up sandbox container '{CONTAINER_NAME}'...")
        subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True, check=False)
        subprocess.run(["docker", "rm", CONTAINER_NAME], capture_output=True, check=False)
        print("[docker_sandbox] Cleanup complete.")

    @classmethod
    def _register_cleanup(cls) -> None:
        if not cls._is_registered:
            atexit.register(cls.stop)
            cls._is_registered = True
