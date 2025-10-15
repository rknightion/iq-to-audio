from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import docker
from docker import DockerClient
from docker.errors import APIError, DockerException
from docker.models.containers import Container

LOG = logging.getLogger(__name__)

DEFAULT_IMAGE = "ghcr.io/rknightion/iq-to-audio-backend:latest"
DEFAULT_AUDIO_TARGET = Path("/workspace/audio")


class DockerBackendError(RuntimeError):
    """Raised for failures while orchestrating the backend container."""


class DockerConnectionError(DockerBackendError):
    """Raised when the Docker engine is unavailable."""


class DockerLaunchError(DockerBackendError):
    """Raised when the backend container could not be launched or completed."""


@dataclass(slots=True)
class DockerBackendConfig:
    """Configuration for the shared backend container image."""

    image: str = DEFAULT_IMAGE
    audio_mount: Path = DEFAULT_AUDIO_TARGET
    environment: dict[str, str] = field(default_factory=dict)
    auto_remove: bool = True
    tty: bool = True
    stdin_open: bool = True


@dataclass(slots=True)
class DockerLaunchRequest:
    """Launch parameters for a backend decoder invocation."""

    command: tuple[str, ...]
    audio_dir: Path
    decoder_key: str = "custom"
    pull_if_missing: bool = True

    def validate(self) -> None:
        if not self.command:
            raise ValueError(
                "Launch request must include a command to execute inside the container."
            )
        if not self.audio_dir.exists():
            raise ValueError(f"Audio directory does not exist: {self.audio_dir}")
        if not self.audio_dir.is_dir():
            raise ValueError(f"Audio path is not a directory: {self.audio_dir}")


@dataclass(slots=True)
class DockerConnectivity:
    available: bool
    message: str


class DockerBackend:
    """Thin wrapper around docker-py used by the CLI and UI."""

    def __init__(
        self,
        *,
        config: DockerBackendConfig | None = None,
        client: DockerClient | None = None,
    ) -> None:
        self.config = config or DockerBackendConfig()
        self._client = client or docker.from_env()

    @property
    def client(self) -> DockerClient:
        return self._client

    def probe(self) -> DockerConnectivity:
        """Check whether the Docker engine is reachable."""
        try:
            self._client.ping()
            # Listing containers exercises permissions beyond ping.
            self._client.containers.list(limit=1)
        except (DockerException, OSError) as exc:
            message = str(exc).strip() or "Unable to communicate with the Docker engine."
            LOG.debug("Docker probe failed: %s", message)
            return DockerConnectivity(False, message)
        return DockerConnectivity(True, "Docker engine reachable.")

    def ensure_connection(self) -> None:
        """Raise if the Docker engine is not reachable."""
        connectivity = self.probe()
        if not connectivity.available:
            raise DockerConnectionError(connectivity.message)

    def pull_image(self) -> None:
        """Ensure the backend image is available locally."""
        try:
            LOG.debug("Pulling backend image %s", self.config.image)
            self._client.images.pull(self.config.image)
        except DockerException as exc:  # pragma: no cover - docker errors vary widely
            message = str(exc).strip() or f"Failed to pull {self.config.image}."
            raise DockerBackendError(message) from exc

    def _volume_spec(self, source: Path) -> dict[str, dict[str, str]]:
        resolved = source.resolve()
        return {
            resolved.as_posix(): {
                "bind": self.config.audio_mount.as_posix(),
                "mode": "rw",
            }
        }

    def run_and_stream(
        self,
        request: DockerLaunchRequest,
        *,
        log_callback: Callable[[str], None] | None = None,
    ) -> int:
        """Run the backend container, streaming stdout/stderr to a callback."""
        request.validate()
        if request.pull_if_missing:
            self._ensure_image_available()
        options = {
            "image": self.config.image,
            "command": list(request.command),
            "detach": True,
            "remove": self.config.auto_remove,
            "volumes": self._volume_spec(request.audio_dir),
            "working_dir": self.config.audio_mount.as_posix(),
            "environment": self.config.environment or None,
            "tty": self.config.tty,
            "stdin_open": self.config.stdin_open,
        }
        LOG.debug(
            "Launching backend container image=%s command=%s audio_dir=%s",
            self.config.image,
            request.command,
            request.audio_dir,
        )
        try:
            container = self._client.containers.run(**options)
        except (DockerException, OSError) as exc:
            message = str(exc).strip() or "Failed to start backend container."
            raise DockerLaunchError(message) from exc

        try:
            for chunk in self._stream_container_logs(container):
                if log_callback is not None:
                    log_callback(chunk)
                else:
                    LOG.info("%s", chunk.rstrip("\n"))
        finally:
            exit_info = self._wait_for_exit(container)
        status_code = self._coerce_status_code(exit_info.get("StatusCode", 1))
        if status_code != 0:
            error = exit_info.get("Error")
            if error:
                message = str(error).strip()
            else:
                message = f"Backend container exited with status {status_code}."
            raise DockerLaunchError(message)
        return status_code

    def _stream_container_logs(self, container: Container) -> Iterable[str]:
        try:
            stream = container.attach(
                stream=True,
                stdout=True,
                stderr=True,
                logs=True,
            )
        except DockerException as exc:
            raise DockerLaunchError(
                str(exc).strip() or "Unable to attach to backend container."
            ) from exc

        for chunk in stream:
            if isinstance(chunk, bytes):
                text = chunk.decode("utf-8", errors="replace")
            else:
                text = str(chunk)
            yield text

    def _wait_for_exit(self, container: Container) -> dict[str, object]:
        try:
            result = container.wait()
        except (DockerException, APIError) as exc:
            raise DockerLaunchError(
                str(exc).strip() or "Failed while waiting for container exit."
            ) from exc
        finally:
            if not self.config.auto_remove:
                with contextlib.suppress(DockerException, APIError):
                    container.remove(force=True)
        return cast(dict[str, object], result)

    def _ensure_image_available(self) -> None:
        try:
            self._client.images.get(self.config.image)
        except DockerException:
            self.pull_image()

    @staticmethod
    def _coerce_status_code(value: object) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return 1
        if isinstance(value, float):
            return int(value)
        return 1


def default_decoder_command(command: Sequence[str] | None = None) -> tuple[str, ...]:
    """Normalize decoder commands to a tuple for downstream use."""
    if command:
        return tuple(command)
    return ("dsd-fme", "--help")
