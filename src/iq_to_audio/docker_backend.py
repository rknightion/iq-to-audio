from __future__ import annotations

import contextlib
import logging
import lzma
import platform
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

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


@dataclass(slots=True)
class DockerImageInfo:
    """Information about the backend Docker image."""

    present: bool
    tags: list[str]
    created: str | None  # ISO format timestamp
    size_mb: float | None

    def format_status(self) -> str:
        """Format a user-friendly status string."""
        if not self.present:
            return "Image not yet loaded"
        tag = self.tags[0] if self.tags else "unknown"
        size_str = f"{self.size_mb:.0f} MB" if self.size_mb else "unknown size"
        created_str = self.created.split("T")[0] if self.created else "unknown date"
        return f"Image: {tag} ({size_str}, built {created_str})"


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

    def get_image_info(self) -> DockerImageInfo:
        """Get information about the backend image."""
        try:
            image = self._client.images.get(self.config.image)
            tags = image.tags if hasattr(image, "tags") else []
            created = image.attrs.get("Created") if hasattr(image, "attrs") else None
            size_bytes = image.attrs.get("Size", 0) if hasattr(image, "attrs") else 0
            size_mb = size_bytes / (1024 * 1024) if size_bytes else None
            return DockerImageInfo(
                present=True,
                tags=tags,
                created=created,
                size_mb=size_mb,
            )
        except DockerException:
            return DockerImageInfo(present=False, tags=[], created=None, size_mb=None)

    def pull_image(self) -> None:
        """Ensure the backend image is available locally."""
        try:
            LOG.debug("Pulling backend image %s", self.config.image)
            self._client.images.pull(self.config.image)
        except DockerException as exc:  # pragma: no cover - docker errors vary widely
            message = str(exc).strip() or f"Failed to pull {self.config.image}."
            raise DockerBackendError(message) from exc

    def load_bundled_image(
        self,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> bool:
        """Load the bundled Docker image tar if available.

        Returns:
            True if the bundled image was loaded successfully, False if no bundled image was found.

        Raises:
            DockerBackendError: If the bundled image exists but failed to load.
        """
        tar_path = self._locate_bundled_tar()
        if tar_path is None:
            LOG.debug("No bundled Docker image found")
            return False

        LOG.info("Loading bundled Docker image from %s", tar_path)
        if progress_callback:
            progress_callback("Extracting bundled container image...")

        # Decompress xz to temporary file
        tmp_tar = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
                tmp_tar = Path(tmp.name)

            with lzma.open(tar_path, "rb") as compressed, open(tmp_tar, "wb") as uncompressed:
                shutil.copyfileobj(compressed, uncompressed)

            if progress_callback:
                progress_callback("Loading container into Docker...")

            # Load the tar into Docker
            with open(tmp_tar, "rb") as f:
                self._client.images.load(f.read())

            LOG.info("Successfully loaded bundled Docker image")
            return True

        except (lzma.LZMAError, OSError) as exc:
            message = f"Failed to decompress bundled image: {exc}"
            LOG.error(message)
            raise DockerBackendError(message) from exc
        except DockerException as exc:
            message = f"Failed to load bundled image into Docker: {exc}"
            LOG.error(message)
            raise DockerBackendError(message) from exc
        finally:
            if tmp_tar is not None:
                with contextlib.suppress(OSError):
                    tmp_tar.unlink()

    @staticmethod
    def _locate_bundled_tar() -> Path | None:
        """Locate the bundled Docker image tar for the current platform."""
        # Determine bundle directory (PyInstaller or dev environment)
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            # PyInstaller sets _MEIPASS at runtime
            bundle_dir = Path(sys._MEIPASS) / "docker"
        else:
            # Development mode: look relative to this file
            bundle_dir = Path(__file__).parent.parent.parent / "packaging" / "docker"

        # Determine platform-specific tar name
        machine = platform.machine().lower()
        if machine in ("x86_64", "amd64"):
            tar_name = "backend-amd64.tar.xz"
        elif machine in ("arm64", "aarch64"):
            tar_name = "backend-arm64.tar.xz"
        else:
            LOG.warning("Unsupported architecture for bundled image: %s", machine)
            return None

        tar_path = bundle_dir / tar_name
        if not tar_path.exists():
            LOG.debug("Bundled Docker image not found at %s", tar_path)
            return None

        return tar_path

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
        """Ensure the backend image is available, trying bundled tar first."""
        try:
            self._client.images.get(self.config.image)
            LOG.debug("Image %s already present in Docker", self.config.image)
            return
        except DockerException:
            pass  # Image not present, try alternatives

        # Try bundled image first (offline-first approach)
        LOG.info("Image not found locally, attempting to load from bundled tar...")
        try:
            if self.load_bundled_image():
                return
        except DockerBackendError:
            LOG.warning("Failed to load bundled image, will try pull instead")

        # Fall back to pull (requires internet)
        LOG.info("Bundled image unavailable, pulling from registry...")
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
