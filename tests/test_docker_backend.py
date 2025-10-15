from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from docker.errors import DockerException

from iq_to_audio.docker_backend import (
    DEFAULT_AUDIO_TARGET,
    DockerBackend,
    DockerBackendConfig,
    DockerLaunchError,
    DockerLaunchRequest,
)


def _build_mock_client(container: MagicMock) -> MagicMock:
    client = MagicMock()
    containers = MagicMock()
    containers.run.return_value = container
    client.containers = containers
    images = MagicMock()
    images.get.return_value = None
    client.images = images
    client.ping.return_value = None
    return client


def test_launch_request_requires_command(tmp_path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    request = DockerLaunchRequest(command=(), audio_dir=audio_dir)
    with pytest.raises(ValueError, match="command"):
        request.validate()


def test_launch_request_requires_directory(tmp_path) -> None:
    missing = tmp_path / "missing"
    request = DockerLaunchRequest(command=("dsd-fme",), audio_dir=missing)
    with pytest.raises(ValueError, match="does not exist"):
        request.validate()


def test_run_and_stream_invokes_docker(tmp_path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    container = MagicMock()
    container.attach.return_value = iter([b"log line 1\n", b"log line 2\n"])
    container.wait.return_value = {"StatusCode": 0}

    client = _build_mock_client(container)
    backend = DockerBackend(config=DockerBackendConfig(image="test-image"), client=client)
    request = DockerLaunchRequest(
        command=("dsd-fme", "--help"),
        audio_dir=audio_dir,
        decoder_key="dsd-fme",
        pull_if_missing=False,
    )

    logs: list[str] = []
    status = backend.run_and_stream(request, log_callback=logs.append)
    assert status == 0
    assert logs == ["log line 1\n", "log line 2\n"]

    run_kwargs = client.containers.run.call_args.kwargs
    assert run_kwargs["image"] == "test-image"
    assert run_kwargs["command"] == ["dsd-fme", "--help"]
    assert (
        run_kwargs["volumes"][audio_dir.resolve().as_posix()]["bind"]
        == DEFAULT_AUDIO_TARGET.as_posix()
    )
    assert run_kwargs["working_dir"] == DEFAULT_AUDIO_TARGET.as_posix()
    container.wait.assert_called_once()


def test_run_and_stream_propagates_exit_code(tmp_path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    container = MagicMock()
    container.attach.return_value = iter([b"stderr\n"])
    container.wait.return_value = {"StatusCode": 42, "Error": "decoder failed"}

    client = _build_mock_client(container)
    backend = DockerBackend(config=DockerBackendConfig(image="test-image"), client=client)
    request = DockerLaunchRequest(
        command=("dsd-fme",),
        audio_dir=audio_dir,
        decoder_key="dsd-fme",
        pull_if_missing=False,
    )

    with pytest.raises(DockerLaunchError, match="decoder failed"):
        backend.run_and_stream(request)


def test_probe_reports_unavailable() -> None:
    container = MagicMock()
    client = _build_mock_client(container)
    client.ping.side_effect = DockerException("boom")
    backend = DockerBackend(client=client)

    status = backend.probe()
    assert not status.available
    assert "boom" in status.message
