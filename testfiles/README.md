# Test Fixtures

Integration tests rely on large IQ waveform captures. To avoid duplicating
hundreds of megabytes in the repository, the captures are packaged in the
`iq-to-audio-fixtures.tar.xz` archive (tracked via Git LFS using the LZMA2
`xz -9e -T0` settings).

To update the archive after modifying the WAV files:

```bash
XZ_OPT='-9e -T0' tar -C testfiles -cJf testfiles/iq-to-audio-fixtures.tar.xz \
  fc-132334577Hz-ft-132300000-AM.wav \
  fc-456834049Hz-ft-455837500-ft2-456872500-NFM.wav
```

The test fixtures in `tests/conftest.py` automatically extract the required
WAV files from the archive when they are missing. The raw `.wav` files remain
ignored by git, so they can be deleted safely after repackaging.
