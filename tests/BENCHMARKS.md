# Performance Benchmarks

This document describes the performance benchmarks tracked over time for the iq-to-audio project.

## Running Benchmarks

```bash
# Run all benchmarks and save results
uv run pytest tests/test_benchmark.py \
  --benchmark-json=benchmark_results.json \
  -p no:pytestqt

# Run and display comparison table
uv run pytest tests/test_benchmark.py \
  --benchmark-only \
  -p no:pytestqt

# Run specific group
uv run pytest tests/test_benchmark.py \
  -m benchmark \
  --benchmark-group=demod-synthetic \
  -p no:pytestqt
```

**Note**: The `-p no:pytestqt` flag disables the Qt plugin, which isn't needed for benchmarks and avoids Qt dependency issues.

## Benchmark Groups

### 1. Demodulation Modes (`demod-synthetic`)

Tests all supported demodulation modes with synthetic signals:
- **NFM** (Narrowband FM) - 12.5 kHz bandwidth, typical for two-way radio
- **AM** (Amplitude Modulation) - 10 kHz bandwidth, typical for aviation/broadcast
- **USB** (Upper Sideband) - 2.8 kHz bandwidth, typical for HF voice
- **LSB** (Lower Sideband) - 2.8 kHz bandwidth, typical for HF voice

**Why track this**: Ensures each demodulation algorithm maintains optimal performance. Regressions here indicate algorithmic issues.

**Baseline**: ~0.5s of 250 kHz signal should process in <50ms on modern hardware.

### 2. Sample Rate Scaling (`sample-rate-scaling`)

Tests processing at different input sample rates:
- **Low** (96 kHz) - Typical for audio applications
- **Medium** (1 MHz) - Typical for RTL-SDR and basic SDR receivers
- **High** (2.5 MHz) - Stress test for high-end SDR hardware

**Why track this**: Validates that performance scales linearly with sample rate. Non-linear scaling indicates inefficiencies in the streaming pipeline.

**Expected scaling**: Processing time should be proportional to sample rate. 2.5 MHz should take ~26× longer than 96 kHz.

### 3. Chunk Size (`chunk-size`)

Tests different chunk sizes for the streaming pipeline:
- **Small** (32k samples) - Lower memory, more function call overhead
- **Medium** (128k samples) - Balanced (default)
- **Large** (512k samples) - Higher memory, less overhead

**Why track this**: Helps identify the optimal memory/performance tradeoff. Large chunks should be faster but use more memory.

**Tuning guide**: If small chunks are nearly as fast as large chunks, overhead is well-optimized. If large chunks are significantly faster, consider increasing default chunk size.

### 4. AGC Impact (`agc-impact`)

Compares performance with and without Automatic Gain Control:
- **With AGC** - Real-world scenario
- **Without AGC** - Baseline computational cost

**Why track this**: AGC adds computational overhead. Track the cost to ensure it remains reasonable (<10% overhead).

**Expected difference**: AGC should add minimal overhead since it operates on decimated audio, not the full IQ stream.

### 5. Bandwidth Scaling (`bandwidth-scaling`)

Tests different signal bandwidths:
- **Narrow** (2.8 kHz) - Voice communications
- **Medium** (12.5 kHz) - Narrowband FM
- **Wide** (200 kHz) - Wideband FM / broadcast

**Why track this**: Wider bandwidths require less decimation, affecting filter design and computational cost.

**Expected behavior**: Wider bandwidths should process faster (less decimation) but may have higher filter complexity.

### 6. Real Files (`real-files`)

Benchmarks using actual captured IQ test files:
- **AM test file** - Real aviation AM signal
- **NFM test file** - Real two-way radio signal

**Why track this**: Synthetic signals may not capture real-world signal characteristics (fading, noise, interference). Real files provide end-to-end validation.

**Note**: These benchmarks require test fixtures to be present. They're skipped if files aren't available.

### 7. Sustained Performance (`sustained-performance`)

Tests processing of longer signal durations:
- **1 second** of 2.5 MHz signal (~5 MB of IQ data)

**Why track this**: Detects memory leaks, cache thrashing, or other issues that only appear with sustained processing.

**Expected behavior**: Time should scale linearly with duration. No slowdown over time indicates good memory management.

## Performance Metrics

Pytest-benchmark tracks:
- **Min time** - Best case (useful for detecting improvements)
- **Max time** - Worst case (useful for detecting regressions)
- **Mean time** - Average (primary tracking metric)
- **Median time** - Typical case (less affected by outliers)
- **Standard deviation** - Consistency (lower is better)
- **Iterations** - Automatically determined for statistical confidence

## Performance Targets

### Current Generation Hardware (2024)

Based on 2.5 MHz sample rate (typical SDR):

| Operation | Target | Notes |
|-----------|--------|-------|
| NFM demod (synthetic) | <50 ms/s | Should achieve >20× realtime |
| AM demod (synthetic) | <40 ms/s | Simpler than FM |
| SSB demod (synthetic) | <35 ms/s | No FM demodulation |
| Real file processing | <100 ms/s | Includes file I/O overhead |

### Realtime Processing

For realtime processing of a 12.5 kHz NFM channel from 2.5 MHz input:
- Processing budget: 1 second per second of signal
- Typical performance: 0.05s processing for 1s signal = 20× realtime
- **Margin**: 19× spare capacity for other operations

## Regression Criteria

A performance regression is defined as:
- **>10% slowdown** in any individual benchmark (alert threshold)
- **>20% slowdown** triggers investigation
- **>50% slowdown** blocks merge

Speedups of >10% should be documented in release notes.

## Viewing Historical Data

Benchmark results are stored in the `benchmarks/` directory (tracked via GitHub Pages):

1. Go to repository Settings → Pages
2. View benchmark trends at: `https://YOUR_ORG.github.io/iq-to-audio/benchmarks/`
3. Compare current vs historical performance

## Adding New Benchmarks

When adding benchmarks:

1. Choose an appropriate group or create a new one
2. Use descriptive test names: `test_benchmark_<operation>_<variant>`
3. Document the purpose in the docstring
4. Add to this README under the appropriate section
5. Ensure benchmarks are deterministic (use fixed RNG seeds)
6. Keep duration reasonable (<2 seconds per benchmark)

Example:

```python
@pytest.mark.benchmark(group="new-feature")
def test_benchmark_new_feature(benchmark):
    """Benchmark new feature with typical parameters."""
    result = benchmark(
        your_function,
        param1=value1,
        param2=value2,
    )
    assert result == expected
```

## Troubleshooting

**Benchmarks are noisy/inconsistent:**
- Ensure machine is idle during benchmark runs
- Increase warmup rounds (pytest-benchmark does this automatically)
- Check for thermal throttling on laptops

**Benchmarks fail in CI:**
- CI runners may be slower/faster than local hardware
- Use relative comparisons (vs. baseline) not absolute thresholds
- GitHub Actions standard runners vary in performance

**Out of memory:**
- Reduce `seconds` parameter in long-duration benchmarks
- Reduce chunk sizes
- Process smaller test files

## References

- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- [GitHub Action for benchmarks](https://github.com/benchmark-action/github-action-benchmark)
