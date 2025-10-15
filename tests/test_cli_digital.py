from __future__ import annotations

from unittest.mock import MagicMock

from iq_to_audio import cli
from iq_to_audio.docker_backend import DockerLaunchError, DockerLaunchRequest


def test_cli_digital_invokes_backend(monkeypatch, tmp_path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    backend = MagicMock()
    backend.ensure_connection.return_value = None
    backend.run_and_stream.return_value = 0

    monkeypatch.setattr(cli, "DockerBackend", lambda config=None: backend)

    result = cli.main(["digital", "--audio-dir", str(audio_dir)])
    assert result == 0

    backend.ensure_connection.assert_called_once()
    backend.run_and_stream.assert_called_once()
    request_arg = backend.run_and_stream.call_args.args[0]
    assert isinstance(request_arg, DockerLaunchRequest)
    assert request_arg.command[0] == "dsd-fme"
    assert request_arg.audio_dir == audio_dir
    assert "log_callback" in backend.run_and_stream.call_args.kwargs


def test_cli_digital_reports_launch_failure(monkeypatch, tmp_path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    backend = MagicMock()
    backend.ensure_connection.return_value = None
    backend.run_and_stream.side_effect = DockerLaunchError("boom")

    monkeypatch.setattr(cli, "DockerBackend", lambda config=None: backend)

    result = cli.main(
        [
            "digital",
            "--audio-dir",
            str(audio_dir),
            "--",
            "multimon-ng",
            "--help",
        ]
    )
    assert result == 1

    backend.ensure_connection.assert_called_once()
    backend.run_and_stream.assert_called_once()
