"""Docker-backed sandbox for isolated execute() calls.

deepagents 0.6.12 (the version pinned in this repo) does not ship a Docker or
Modal sandbox backend - `deepagents.backends.sandbox` only exposes the
abstract `BaseSandbox`. This module implements the missing piece the same way
`deepagents.backends.local_shell.LocalShellBackend` does (extend
`FilesystemBackend` for file ops, add `execute()` yourself), except commands
run inside a long-lived container via `docker exec` instead of directly on
the host.

File ops (read/write/edit/ls/glob/grep) are inherited from `FilesystemBackend`
and operate on the host path directly - this only works because the container
bind-mounts that same directory (see docker-compose.yml: `./run_artifacts ->
/workspace`), so host and container agree on file contents without needing
`docker cp`.

Prerequisite: the container must already be running (`docker compose up -d`
in this repo's `docker/` directory) - this class does not start it for you,
since silently launching containers from inside an agent run is a bigger
blast radius than failing with a clear error.
"""

from __future__ import annotations

import subprocess
import uuid

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

DEFAULT_EXECUTE_TIMEOUT = 1800


class DockerSandbox(FilesystemBackend, SandboxBackendProtocol):
    """Filesystem access on the host + shell execution inside a Docker container.

    Args:
        container_name: Name of the already-running container (matches
            `docker-compose.yml`'s `container_name`).
        root_dir: Host directory bind-mounted into the container. File
            operations (read/write/edit/ls) happen here directly.
        workdir: Working directory *inside* the container to `cd` into
            before running each command (the container-side path that
            `root_dir` is mounted at).
        timeout: Default command timeout in seconds.
    """

    def __init__(
        self,
        *,
        container_name: str,
        root_dir: str,
        workdir: str = "/workspace",
        timeout: int = DEFAULT_EXECUTE_TIMEOUT,
    ) -> None:
        super().__init__(root_dir=root_dir, virtual_mode=True, max_file_size_mb=10)
        self._container_name = container_name
        self._workdir = workdir
        self._default_timeout = timeout
        self._sandbox_id = f"docker-{container_name}-{uuid.uuid4().hex[:8]}"

    @property
    def id(self) -> str:
        return self._sandbox_id

    def _ensure_running(self) -> str | None:
        """Return an error message if the container isn't up, else None."""
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", self._container_name],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except FileNotFoundError:
            return "Error: `docker` CLI not found on PATH. Install Docker Desktop, or use SANDBOX_BACKEND=local instead."
        if result.returncode != 0 or result.stdout.strip() != "true":
            return (
                f"Error: container '{self._container_name}' is not running. "
                "Start it first: `docker compose -f docker/docker-compose.yml up -d`"
            )
        return None

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        not_running = self._ensure_running()
        if not_running is not None:
            return ExecuteResponse(output=not_running, exit_code=1, truncated=False)

        effective_timeout = timeout if timeout is not None else self._default_timeout
        docker_cmd = [
            "docker",
            "exec",
            "-w",
            self._workdir,
            self._container_name,
            "sh",
            "-c",
            command,
        ]
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ExecuteResponse(
                output=f"Error: command timed out after {effective_timeout}s inside container '{self._container_name}'.",
                exit_code=124,
                truncated=False,
            )
        except Exception as exc:  # noqa: BLE001 - surface as ExecuteResponse, not a raised exception
            return ExecuteResponse(output=f"Error executing in container: {exc}", exit_code=1, truncated=False)

        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.extend(f"[stderr] {line}" for line in result.stderr.strip().split("\n"))
        output = "\n".join(output_parts) if output_parts else "<no output>"
        if result.returncode != 0:
            output = f"{output.rstrip()}\n\nExit code: {result.returncode}"
        return ExecuteResponse(output=output, exit_code=result.returncode, truncated=False)

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        import asyncio

        return await asyncio.to_thread(self.execute, command, timeout=timeout)


__all__ = ["DEFAULT_EXECUTE_TIMEOUT", "DockerSandbox"]
