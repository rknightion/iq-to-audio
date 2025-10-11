import math
import signal
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PySide6 import QtWidgets

import iq_to_audio.interactive.app as app_module
from iq_to_audio.interactive import (
    InteractiveWindow,
    SnapshotData,
    StatusProgressSink,
    _SigintRelay,
)
from iq_to_audio.probe import SampleRateProbe
from iq_to_audio.utils import CenterFrequencyResult


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
        assert app.targets_panel is not None
        assert app.targets_panel.clear_button is not None
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


def test_demod_mode_toggles_agc(qapp):
    app = make_app()
    try:
        assert app.recording_panel is not None
        assert app.demod_panel is not None
        agc = app.recording_panel.agc_check
        mode_combo = app.demod_panel.mode_combo
        assert agc.isEnabled()
        none_index = mode_combo.findData("none")
        assert none_index >= 0
        mode_combo.setCurrentIndex(none_index)
        QtWidgets.QApplication.processEvents()
        assert not agc.isEnabled()
        assert not agc.isChecked()
        nfm_index = mode_combo.findData("nfm")
        mode_combo.setCurrentIndex(nfm_index)
        QtWidgets.QApplication.processEvents()
        assert agc.isEnabled()
        assert agc.isChecked()
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


def test_insert_target_frequency_populates_next_slot(qapp):
    app = make_app()
    try:
        assert app.targets_panel is not None
        freqs = [100_000_000.0, 101_000_000.0, 102_000_000.0, 103_000_000.0, 104_000_000.0]
        for idx, freq in enumerate(freqs):
            assert app._insert_target_frequency(freq, announce=False)
            QtWidgets.QApplication.processEvents()
            assert app.targets_panel.entries[idx].text() == f"{freq:.0f}"
        assert app.state.target_freqs == freqs
        assert not app._insert_target_frequency(105_000_000.0, announce=False)
        assert app.state.target_freqs == freqs
    finally:
        app.close()


def test_clear_targets_resets_entries_and_state(qapp):
    app = make_app()
    try:
        assert app.targets_panel is not None
        assert app._insert_target_frequency(100_000_000.0, announce=False)
        assert app.state.target_freqs
        app._on_clear_targets()
        QtWidgets.QApplication.processEvents()
        assert app.state.target_freqs == []
        assert all(entry.text() == "" for entry in app.targets_panel.entries)
    finally:
        app.close()


def test_scroll_event_zoom_behavior(qapp):
    app = make_app()
    try:
        rng = np.random.default_rng(12345)
        snapshot = SnapshotData(
            path=Path("dummy.wav"),
            sample_rate=48_000.0,
            center_freq=100_000_000.0,
            probe=SampleRateProbe(ffprobe=48_000.0, header=48_000.0),
            seconds=1.0,
            mode="samples",
            freqs=np.linspace(-50_000.0, 50_000.0, 256),
            psd_db=rng.uniform(-90.0, -10.0, 256),
            waterfall=None,
            samples=None,
            params={},
            fft_frames=1,
        )
        app._render_snapshot(snapshot, remember=True)
        QtWidgets.QApplication.processEvents()
        assert app.ax_main is not None
        initial_xlim = app.ax_main.get_xlim()
        center = sum(initial_xlim) / 2.0
        event_in = types.SimpleNamespace(step=1.0, inaxes=app.ax_main, xdata=center)
        app._on_canvas_scroll(event_in)
        QtWidgets.QApplication.processEvents()
        zoom_in_xlim = app.ax_main.get_xlim()
        assert zoom_in_xlim[1] - zoom_in_xlim[0] < initial_xlim[1] - initial_xlim[0]
        event_out = types.SimpleNamespace(step=-1.0, inaxes=app.ax_main, xdata=center)
        app._on_canvas_scroll(event_out)
        QtWidgets.QApplication.processEvents()
        zoom_out_xlim = app.ax_main.get_xlim()
        assert zoom_out_xlim[1] - zoom_out_xlim[0] > zoom_in_xlim[1] - zoom_in_xlim[0]
    finally:
        app.close()


def test_scroll_zoom_clamps_to_bandwidth(qapp):
    app = make_app()
    try:
        rng = np.random.default_rng(54321)
        snapshot = SnapshotData(
            path=Path("dummy.wav"),
            sample_rate=96_000.0,
            center_freq=144_000_000.0,
            probe=SampleRateProbe(ffprobe=96_000.0, header=96_000.0),
            seconds=1.0,
            mode="samples",
            freqs=np.linspace(-80_000.0, 80_000.0, 512),
            psd_db=rng.uniform(-100.0, -20.0, 512),
            waterfall=None,
            samples=None,
            params={},
            fft_frames=1,
        )
        app._render_snapshot(snapshot, remember=True)
        QtWidgets.QApplication.processEvents()
        assert app.ax_main is not None
        freq_min = float(snapshot.center_freq + np.min(snapshot.freqs))
        freq_max = float(snapshot.center_freq + np.max(snapshot.freqs))
        center = sum(app.ax_main.get_xlim()) / 2.0
        for _ in range(12):
            event = types.SimpleNamespace(step=-5.0, inaxes=app.ax_main, xdata=center)
            app._on_canvas_scroll(event)
        QtWidgets.QApplication.processEvents()
        final_xlim = app.ax_main.get_xlim()
        assert math.isclose(final_xlim[0], freq_min, rel_tol=1e-6, abs_tol=1e-3)
        assert math.isclose(final_xlim[1], freq_max, rel_tol=1e-6, abs_tol=1e-3)
    finally:
        app.close()


def test_detect_button_overrides_manual_when_idle(qapp, tmp_path, monkeypatch):
    app = make_app()
    try:
        def fake_auto_detect_format(self, path, *, announce):
            return None

        monkeypatch.setattr(app, "_auto_detect_format", types.MethodType(fake_auto_detect_format, app))

        def fake_detect(_path):
            return CenterFrequencyResult(200_000_000.0, "mock:button")

        monkeypatch.setattr(app_module, "detect_center_frequency", fake_detect)

        recording = tmp_path / "recording.wav"
        recording.touch()
        app._set_selected_path(recording)
        QtWidgets.QApplication.processEvents()
        assert app.recording_panel is not None
        assert app.recording_panel.detect_button.isEnabled()

        app.recording_panel.center_entry.setText("180000000")
        app._on_center_manual()
        assert app.state.center_source == "manual"
        app._on_detect_center()
        QtWidgets.QApplication.processEvents()
        assert app.state.center_source == "mock:button"
        assert app.state.center_freq == pytest.approx(200_000_000.0)
        assert app.recording_panel.center_entry.text() == "200000000"
    finally:
        app.close()


def test_detect_button_preserves_manual_on_failure(qapp, tmp_path, monkeypatch):
    app = make_app()
    try:
        def fake_auto_detect_format(self, path, *, announce):
            return None

        monkeypatch.setattr(app, "_auto_detect_format", types.MethodType(fake_auto_detect_format, app))

        def detect_success(_path):
            return CenterFrequencyResult(160_000_000.0, "mock:first")

        monkeypatch.setattr(app_module, "detect_center_frequency", detect_success)

        recording = tmp_path / "recording.wav"
        recording.touch()
        app._set_selected_path(recording)
        QtWidgets.QApplication.processEvents()
        assert app.state.center_source == "mock:first"
        assert app.state.center_freq == pytest.approx(160_000_000.0)

        assert app.recording_panel is not None
        app.recording_panel.center_entry.setText("180000000")
        app._on_center_manual()
        assert app.state.center_source == "manual"
        assert app.state.center_freq == pytest.approx(180_000_000.0)

        def detect_failure(_path):
            return CenterFrequencyResult(None, "mock:none")

        monkeypatch.setattr(app_module, "detect_center_frequency", detect_failure)
        app._on_detect_center()
        QtWidgets.QApplication.processEvents()
        assert app.state.center_source == "manual"
        assert app.state.center_freq == pytest.approx(180_000_000.0)
        assert app.recording_panel.center_entry.text() == "180000000"
    finally:
        app.close()


def test_new_file_detection_runs_even_after_manual_override(qapp, tmp_path, monkeypatch):
    app = make_app()
    try:
        def fake_auto_detect_format(self, path, *, announce):
            return None

        monkeypatch.setattr(app, "_auto_detect_format", types.MethodType(fake_auto_detect_format, app))

        path1 = tmp_path / "recording1.wav"
        path1.touch()
        path2 = tmp_path / "recording2.wav"
        path2.touch()

        mapping = {
            path1.resolve(): CenterFrequencyResult(150_000_000.0, "mock:first"),
            path2.resolve(): CenterFrequencyResult(250_000_000.0, "mock:second"),
        }

        def fake_detect(path):
            return mapping.get(Path(path).resolve(), CenterFrequencyResult(None, "mock:none"))

        monkeypatch.setattr(app_module, "detect_center_frequency", fake_detect)

        app._set_selected_path(path1)
        QtWidgets.QApplication.processEvents()
        assert app.state.center_source == "mock:first"
        assert app.state.center_freq == pytest.approx(150_000_000.0)

        assert app.recording_panel is not None
        app.recording_panel.center_entry.setText("555000000")
        app._on_center_manual()
        assert app.state.center_source == "manual"
        assert app.state.center_freq == pytest.approx(555_000_000.0)

        app._set_selected_path(path2)
        QtWidgets.QApplication.processEvents()
        assert app.state.center_source == "mock:second"
        assert app.state.center_freq == pytest.approx(250_000_000.0)
        assert app.recording_panel.center_entry.text() == "250000000"
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
