import pytest

from PySide6 import QtWidgets

from iq_to_audio.interactive import StatusProgressSink, _InteractiveApp


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def make_app():
    base_kwargs = {
        "demod_mode": "nfm",
        "squelch_enabled": True,
        "silence_trim": False,
        "agc_enabled": True,
        "center_freq": None,
        "bandwidth": 12_500.0,
    }
    return _InteractiveApp(
        base_kwargs=base_kwargs,
        initial_path=None,
        snapshot_seconds=1.0,
    )


def test_window_initialization(qapp):
    app = make_app()
    try:
        assert app.windowTitle() == "IQ to Audio — Interactive Mode"
        assert isinstance(app.centralWidget(), QtWidgets.QScrollArea)
        assert app.status_label is not None
        assert app.preview_btn is not None
    finally:
        app.close()


def test_status_signal_updates_label(qapp):
    app = make_app()
    try:
        app.status_update_signal.emit("Testing status", False)
        QtWidgets.QApplication.processEvents()
        assert app.status_label is not None
        assert app.status_label.text() == "Testing status"
    finally:
        app.close()


def test_status_progress_sink_emits_via_signal(qapp):
    app = make_app()
    try:
        messages = []
        sink = StatusProgressSink(lambda msg, err: messages.append((msg, err)))
        sink.status("Working")
        sink.close()
        assert ("Working", True) in messages
        assert ("Processing complete.", False) in messages
    finally:
        app.close()
