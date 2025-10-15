from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PySide6 import QtCore

from ..input_formats import resolve_input_format
from ..preview import run_preview
from ..probe import SampleRateProbe, probe_sample_rate
from ..processing import ProcessingCancelled, ProcessingConfig, tune_chunk_size
from ..spectrum import WaterfallResult, streaming_waterfall
from ..squelch import AudioPostOptions, process_audio_batch
from ..utils import detect_center_frequency
from .models import SnapshotData


def waterfall_to_tuple(
    waterfall: WaterfallResult | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if waterfall is None:
        return None
    return (
        np.asarray(waterfall.freqs, dtype=np.float64),
        np.asarray(waterfall.times, dtype=np.float32),
        np.asarray(waterfall.matrix, dtype=np.float32),
    )


def gather_snapshot(
    config: ProcessingConfig,
    seconds: float,
    *,
    nfft: int,
    hop: int | None,
    max_slices: int,
    fft_workers: int | None,
    max_in_memory_samples: int,
    progress_cb: Callable[[float, float], None] | None = None,
) -> SnapshotData:
    """Stream a preview segment of IQ data for interactive spectrum display."""
    input_spec, _source = resolve_input_format(
        config.in_path,
        requested=config.input_format,
        container_hint=config.input_container,
    )
    manual_rate = config.input_sample_rate
    if manual_rate is not None and manual_rate <= 0:
        raise ValueError("Input sample rate override must be positive.")
    if input_spec.container == "raw":
        if manual_rate is None:
            raise ValueError("Raw IQ inputs require a sample rate override before previewing.")
        sample_rate = float(manual_rate)
        probe = SampleRateProbe(ffprobe=None, header=None, wave=sample_rate)
    else:
        probe = probe_sample_rate(config.in_path)
        sample_rate = float(manual_rate) if manual_rate is not None else probe.value

    center_freq = config.center_freq
    if center_freq is None:
        detection = detect_center_frequency(config.in_path)
        if detection.value is None:
            raise ValueError(
                "Center frequency not provided and could not be inferred from WAV metadata or filename. "
                "Provide --fc or enter a value before previewing."
            )
        center_freq = detection.value

    total_samples = int(max(1, round(sample_rate * seconds)))
    hop = max(1, hop or nfft // 4)
    tuned_chunk = tune_chunk_size(sample_rate, config.chunk_size)
    chunk_size = max(tuned_chunk, nfft)
    retain = min(max_in_memory_samples, total_samples)
    retain_buffer = np.empty(retain, dtype=np.complex64) if retain > 0 else None
    retain_pos = 0
    consumed = 0
    last_report = -1.0

    from ..processing import IQReader  # Local import to avoid circular refs.

    def _chunk_iter() -> Iterator[np.ndarray]:
        nonlocal retain_pos, consumed, last_report
        remaining = total_samples
        reader_sample_rate = sample_rate if input_spec.container == "raw" else None
        with IQReader(
            config.in_path,
            chunk_size,
            config.iq_order,
            input_spec,
            sample_rate=reader_sample_rate,
        ) as reader:
            for block in reader:
                if remaining <= 0:
                    break
                use = block
                if block.size > remaining:
                    use = block[:remaining]
                remaining -= use.size
                consumed += use.size
                if retain_buffer is not None and retain_pos < retain_buffer.size:
                    take = min(retain_buffer.size - retain_pos, use.size)
                    if take > 0:
                        retain_buffer[retain_pos : retain_pos + take] = use[:take]
                        retain_pos += take
                if progress_cb:
                    with contextlib.suppress(Exception):
                        frac = min(consumed / total_samples, 1.0)
                        if frac - last_report >= 0.02 or frac >= 0.999:
                            seconds_done = consumed / sample_rate
                            progress_cb(seconds_done, frac)
                            last_report = frac
                yield use
                if remaining <= 0:
                    break

    freqs, avg_psd, waterfall, frames = streaming_waterfall(
        _chunk_iter(),
        sample_rate,
        nfft=nfft,
        hop=hop,
        max_slices=max_slices,
        fft_workers=fft_workers,
    )

    samples = None
    if retain_buffer is not None and retain_pos > 0:
        samples = retain_buffer[:retain_pos].copy()

    params: dict[str, Any] = {
        "nfft": nfft,
        "hop": hop,
        "max_slices": max_slices,
        "fft_workers": fft_workers,
        "seconds": seconds,
        "full_capture": False,
        "max_in_memory_samples": max_in_memory_samples,
    }
    snapshot = SnapshotData(
        path=config.in_path,
        sample_rate=sample_rate,
        center_freq=center_freq,
        probe=probe,
        seconds=consumed / sample_rate,
        mode="samples" if samples is not None else "precomputed",
        freqs=freqs,
        psd_db=avg_psd,
        waterfall=waterfall_to_tuple(waterfall),
        samples=samples,
        params=params,
        fft_frames=frames,
    )
    if progress_cb:
        with contextlib.suppress(Exception):
            progress_cb(snapshot.seconds, 1.0)
    return snapshot


def compute_full_psd(
    config: ProcessingConfig,
    *,
    nfft: int,
    hop: int,
    max_slices: int,
    fft_workers: int | None,
    status_cb: Callable[[str], None] | None = None,
) -> SnapshotData:
    input_spec, _source = resolve_input_format(
        config.in_path,
        requested=config.input_format,
        container_hint=config.input_container,
    )
    manual_rate = config.input_sample_rate
    if manual_rate is not None and manual_rate <= 0:
        raise ValueError("Input sample rate override must be positive.")
    if input_spec.container == "raw":
        if manual_rate is None:
            raise ValueError("Raw IQ inputs require a sample rate override before previewing.")
        sample_rate = float(manual_rate)
        probe = SampleRateProbe(ffprobe=None, header=None, wave=sample_rate)
    else:
        probe = probe_sample_rate(config.in_path)
        sample_rate = float(manual_rate) if manual_rate is not None else probe.value
    center_freq = config.center_freq
    if center_freq is None:
        detection = detect_center_frequency(config.in_path)
        if detection.value is None:
            raise ValueError(
                "Center frequency not provided and could not be inferred from WAV metadata or filename. "
                "Enter a value before using full-record preview."
            )
        center_freq = detection.value

    chunk_samples = max(config.chunk_size, nfft)
    consumed = 0
    try:
        file_size = config.in_path.stat().st_size
    except OSError:
        file_size = 0
    header_bytes = 44 if input_spec.container == "wav" else 0
    frame_bytes = input_spec.bytes_per_frame
    payload_bytes = max(file_size - header_bytes, 0)
    estimated_total_samples = payload_bytes // frame_bytes if payload_bytes > 0 else 0
    estimated_chunks = int(math.ceil(estimated_total_samples / chunk_samples)) if estimated_total_samples else 0
    status_stride = max(1, estimated_chunks // 25) if estimated_chunks else 4

    from ..processing import IQReader

    if status_cb:
        with contextlib.suppress(Exception):
            status_cb("Reading full recording for spectrum analysis…")

    def _chunk_iter() -> Iterator[np.ndarray]:
        nonlocal consumed
        chunk_index = 0
        reader_sample_rate = sample_rate if input_spec.container == "raw" else None
        with IQReader(
            config.in_path,
            chunk_samples,
            config.iq_order,
            input_spec,
            sample_rate=reader_sample_rate,
        ) as reader:
            for block in reader:
                if block is None or block.size == 0:
                    break
                consumed += block.size
                chunk_index += 1
                if status_cb and (
                    chunk_index == 1
                    or chunk_index % status_stride == 0
                    or (estimated_chunks and chunk_index >= estimated_chunks)
                ):
                    with contextlib.suppress(Exception):
                        if estimated_chunks:
                            pct = min(chunk_index / estimated_chunks, 1.0) * 100.0
                            seconds_done = consumed / sample_rate if sample_rate > 0 else 0.0
                            total_seconds = (
                                estimated_total_samples / sample_rate if sample_rate > 0 else 0.0
                            )
                            status_cb(
                                f"Averaging PSD chunk {chunk_index}/{estimated_chunks} "
                                f"({pct:4.1f}% ≈ {seconds_done:.1f}s/{total_seconds:.1f}s)"
                            )
                        else:
                            status_cb(f"Averaging PSD chunk {chunk_index}…")
                yield block

    freqs, avg_psd, waterfall, frames = streaming_waterfall(
        _chunk_iter(),
        sample_rate,
        nfft=nfft,
        hop=hop,
        max_slices=max_slices,
        fft_workers=fft_workers,
    )

    params: dict[str, Any] = {
        "nfft": nfft,
        "hop": hop,
        "max_slices": max_slices,
        "fft_workers": fft_workers,
        "seconds": consumed / sample_rate if sample_rate > 0 else 0.0,
        "full_capture": True,
    }
    return SnapshotData(
        path=config.in_path,
        sample_rate=sample_rate,
        center_freq=center_freq,
        probe=probe,
        seconds=consumed / sample_rate if sample_rate > 0 else 0.0,
        mode="precomputed",
        freqs=freqs,
        psd_db=avg_psd,
        waterfall=waterfall_to_tuple(waterfall),
        samples=None,
        params=params,
        fft_frames=frames,
    )


class WorkerSignals(QtCore.QObject):
    """Qt-backed signals shared by background workers."""

    started = QtCore.Signal()
    progress = QtCore.Signal(float, float)
    failed = QtCore.Signal(Exception)
    finished = QtCore.Signal(object)


@dataclass(slots=True)
class SnapshotJob:
    config: ProcessingConfig
    seconds: float
    nfft: int
    hop: int | None
    max_slices: int
    fft_workers: int | None
    max_samples: int
    full_capture: bool = False


class SnapshotWorker(QtCore.QRunnable):
    """QRunnable-based worker that gathers a snapshot in a thread pool."""

    def __init__(self, job: SnapshotJob) -> None:
        super().__init__()
        self.job = job
        self.signals = WorkerSignals()

    @QtCore.Slot()
    def run(self) -> None:
        self.signals.started.emit()
        try:
            if self.job.full_capture:
                snapshot = compute_full_psd(
                    self.job.config,
                    nfft=self.job.nfft,
                    hop=self.job.hop or max(1, self.job.nfft // 4),
                    max_slices=self.job.max_slices,
                    fft_workers=self.job.fft_workers,
                    status_cb=None,
                )
            else:
                snapshot = gather_snapshot(
                    self.job.config,
                    self.job.seconds,
                    nfft=self.job.nfft,
                    hop=self.job.hop,
                    max_slices=self.job.max_slices,
                    fft_workers=self.job.fft_workers,
                    max_in_memory_samples=self.job.max_samples,
                    progress_cb=self.signals.progress.emit,
                )
        except Exception as exc:  # pragma: no cover - worker surfaces error upstream
            self.signals.failed.emit(exc)
        else:
            self.signals.finished.emit(snapshot)


class PreviewWorker(QtCore.QRunnable):
    """Run preview DSP pipelines for one or more configs."""

    def __init__(
        self,
        configs: list[ProcessingConfig],
        seconds: float,
        *,
        progress_sink,
        register_pipeline: Callable[[object | None], None],
    ) -> None:
        super().__init__()
        self.configs = configs
        self.seconds = seconds
        self.progress_sink = progress_sink
        self.register_pipeline = register_pipeline
        self.signals = WorkerSignals()

    @QtCore.Slot()
    def run(self) -> None:
        self.signals.started.emit()
        outputs: list[Path] = []
        total = len(self.configs)
        current_pipeline = None

        def _capture_pipeline(pipeline):
            nonlocal current_pipeline
            current_pipeline = pipeline

        try:
            for index, config in enumerate(self.configs, start=1):
                if total > 1:
                    with contextlib.suppress(Exception):
                        self.progress_sink.status(f"Preview {index}/{total}")

                # Clear previous pipeline reference
                current_pipeline = None

                _, preview_path = run_preview(
                    config,
                    self.seconds,
                    progress_sink=self.progress_sink,
                    on_pipeline=_capture_pipeline,
                )
                outputs.append(preview_path)

                # Clear pipeline after successful completion
                current_pipeline = None
        except ProcessingCancelled as exc:
            self.signals.failed.emit(exc)
        except Exception as exc:  # pragma: no cover - worker surfaces error upstream
            self.signals.failed.emit(exc)
        else:
            self.signals.finished.emit(outputs)
        finally:
            # Ensure active pipeline is cancelled and cleaned up
            if current_pipeline is not None:
                with contextlib.suppress(Exception):
                    current_pipeline.cancel()

            # Notify that no pipeline is active
            with contextlib.suppress(Exception):
                self.register_pipeline(None)


@dataclass(slots=True)
class AudioPostJob:
    targets: list[Path]
    options: AudioPostOptions


class AudioPostWorker(QtCore.QRunnable):
    """Background worker for audio post-processing squelch cleanup."""

    def __init__(self, job: AudioPostJob) -> None:
        super().__init__()
        self.job = job
        self.signals = WorkerSignals()

    @QtCore.Slot()
    def run(self) -> None:
        self.signals.started.emit()

        def _progress(done: int, total: int, _path: Path) -> None:
            self.signals.progress.emit(float(done), float(total))

        try:
            summary = process_audio_batch(self.job.targets, self.job.options, progress_cb=_progress)
        except Exception as exc:  # pragma: no cover - surfaced to UI
            self.signals.failed.emit(exc)
        else:
            self.signals.finished.emit(summary)
