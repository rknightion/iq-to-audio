import signal
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PySide6 import QtWidgets

from iq_to_audio.interactive import (
    InteractiveWindow,
    SnapshotData,
    StatusProgressSink,
    _SigintRelay,
)
from iq_to_audio.probe import SampleRateProbe


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def make_app():
    base_kwargs = {
        "demod_mode": "nfm",
        "agc_enabled": True,
        "center_freq": None,
        "bandwidth": 12_500.0,
    }
    return InteractiveWindow(
        base_kwargs=base_kwargs,
        initial_path=None,
        snapshot_seconds=1.0,
    )


def test_window_initialization(qapp):
    app = make_app()
    try:
        QtWidgets.QApplication.processEvents()
        assert app.windowTitle() == "IQ to Audio — Interactive Mode"
        assert isinstance(app.centralWidget(), QtWidgets.QSplitter)
        assert app.toolbar_widget is not None
        assert app.status_panel is not None
        assert app.status_panel.status_label is not None
        assert app.status_panel.preview_button is not None
        assert not app.status_panel.preview_button.isEnabled()
        assert app.status_panel.progress_bar is not None
        assert not app.status_panel.progress_bar.isVisible()
        assert app.main_splitter is not None
        assert len(app.main_splitter.sizes()) == 2
        assert sum(app.main_splitter.sizes()) > 0
        assert app.preview_splitter is not None
        assert len(app.preview_splitter.sizes()) == 2
        assert sum(app.preview_splitter.sizes()) > 0
        assert app.recording_panel is not None
        assert app.recording_panel.format_combo.count() >= 1
        assert app.channel_panel is not None
        assert app.channel_panel.sample_rate_entry is not None
    finally:
        app.close()


def test_status_signal_updates_label(qapp):
    app = make_app()
    try:
        app.status_update_signal.emit("Testing status", False)
        QtWidgets.QApplication.processEvents()
        assert app.status_panel is not None
        assert app.status_panel.status_label.text() == "Testing status"
    finally:
        app.close()


def test_format_combo_manual_override(qapp, tmp_path):
    app = make_app()
    try:
        path = tmp_path / "sample.wav"
        data = np.zeros((32, 2), dtype=np.float32)
        sf.write(path, data, samplerate=48_000, subtype="PCM_16")
        app._set_selected_path(path)
        QtWidgets.QApplication.processEvents()
        assert app.recording_panel is not None
        combo = app.recording_panel.format_combo
        assert combo.itemData(combo.currentIndex()) == "auto"
        manual_index = combo.findData("wav:pcm_s16le")
        assert manual_index >= 0
        combo.setCurrentIndex(manual_index)
        QtWidgets.QApplication.processEvents()
        assert app.state.input_format_choice == "wav:pcm_s16le"
        assert app.state.base_kwargs.get("input_format") == "pcm_s16le"
        assert app.state.base_kwargs.get("input_format_source") == "manual"
        auto_index = combo.findData("auto")
        assert auto_index >= 0
        combo.setCurrentIndex(auto_index)
        QtWidgets.QApplication.processEvents()
        assert app.state.input_format_choice == "auto"
    finally:
        app.close()


def test_status_progress_sink_emits_via_signal(qapp):
    app = make_app()
    try:
        messages = []
        progress = []
        sink = StatusProgressSink(
            lambda msg, err: messages.append((msg, err)),
            progress_update=lambda ratio: progress.append(ratio),
        )
        sink.start([], overall_total=4.0)
        sink.advance("phase", 1.0, overall_completed=1.0, overall_total=4.0)
        sink.status("Working")
        sink.close()
        assert any(msg.startswith("Working") and flag for msg, flag in messages)
        assert any(msg.startswith("Processing complete.") and not flag for msg, flag in messages)
        assert progress[0] == pytest.approx(0.0)
        assert progress[-1] == pytest.approx(1.0)
        assert any(abs(val - 0.25) < 1e-6 for val in progress)
    finally:
        app.close()


def test_preview_action_syncs_with_button(qapp):
    app = make_app()
    try:
        assert app.status_panel is not None
        preview_button = app.status_panel.preview_button
        assert preview_button is not None
        app.state.snapshot_data = SnapshotData(
            path=Path("dummy.wav"),
            sample_rate=48_000.0,
            center_freq=100_000_000.0,
            probe=SampleRateProbe(ffprobe=48_000.0, header=48_000.0),
            seconds=1.0,
            mode="samples",
            freqs=np.array([0.0]),
            psd_db=np.array([0.0]),
            waterfall=None,
            samples=None,
            params={},
            fft_frames=1,
        )
        app._update_status_controls()
        app._set_preview_enabled(True)
        assert preview_button.isEnabled()
        toolbar_preview = next(
            (action for action in app.toolbar_widget.actions() if action.text().startswith("Preview")), None
        )
        assert toolbar_preview is not None
        assert toolbar_preview.isEnabled()
        app._set_preview_enabled(False)
        assert not preview_button.isEnabled()
        assert not toolbar_preview.isEnabled()
    finally:
        app.close()


def test_agc_toggle_updates_state(qapp):
    app = make_app()
    try:
        rp = app.recording_panel
        assert rp is not None
        agc = rp.agc_check
        assert agc is not None
        assert agc.isChecked()
        agc.click()
        QtWidgets.QApplication.processEvents()
        assert not agc.isChecked()
        assert not app.state.agc_enabled
        assert app.state.base_kwargs.get("agc_enabled") is False
        agc.click()
        QtWidgets.QApplication.processEvents()
        assert agc.isChecked()
        assert app.state.agc_enabled
        assert app.state.base_kwargs.get("agc_enabled") is True
    finally:
        app.close()


def test_sigint_relay_installs_and_restores(monkeypatch):
    calls = []

    def fake_signal(sig, handler):
        calls.append((sig, handler))

    monkeypatch.setattr(signal, "signal", fake_signal)
    previous = object()
    relay = _SigintRelay(
        app=None,
        previous_handler=previous,
        schedule_quit=lambda: None,
        escalate=lambda *_: None,
    )
    relay.install()
    assert calls[-1] == (signal.SIGINT, relay)
    relay.restore()
    assert calls[-1] == (signal.SIGINT, previous)


def test_sigint_relay_quits_then_escalates():
    events = []

    relay = _SigintRelay(
        app=None,
        previous_handler=lambda *args: events.append(("previous", args)),
        schedule_quit=lambda: events.append(("quit", None)),
        escalate=lambda signum, frame: events.append(("escalate", signum)),
    )

    relay(signal.SIGINT, None)
    assert ("quit", None) in events
    relay(signal.SIGINT, None)
    assert ("escalate", signal.SIGINT) in events


def test_sigint_relay_default_escalation_calls_previous():
    calls = []

    def previous(signum, frame):
        calls.append((signum, frame))

    relay = _SigintRelay(
        app=None,
        previous_handler=previous,
        schedule_quit=lambda: None,
    )

    relay(signal.SIGINT, "first")
    relay(signal.SIGINT, "second")
    assert calls == [(signal.SIGINT, "second")]
