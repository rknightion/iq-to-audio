# Test Fixtures

Integration tests rely on large IQ waveform captures. To avoid duplicating
hundreds of megabytes in the repository, the captures are packaged in the
`iq-to-audio-fixtures.tar.xz` archive (compressed using LZMA2 `xz -9e -T0`).

## Storage and Distribution

The archive is stored in **Google Drive** instead of Git LFS to:
- Avoid GitHub Actions LFS bandwidth limits and rate limiting
- Enable faster downloads via Google Drive's CDN
- Reduce CI build times

The archive is automatically downloaded:
- In **CI/CD**: via rclone in GitHub Actions workflows
- **Locally**: via pytest session fixture when running tests

See [`docs/GOOGLE_DRIVE_SETUP.md`](../docs/GOOGLE_DRIVE_SETUP.md) for configuration instructions.

## Usage

### Running Tests

Tests automatically download and extract fixtures when needed:

```bash
# Fixtures will be downloaded if not present
uv run pytest

# Or manually download first
python scripts/download_test_fixtures.py
```

**Note**: Local test runs require Google Drive credentials (set via environment variables)
or the archive file must already be present.

### Updating the Archive

To regenerate the archive after modifying WAV files:

```bash
XZ_OPT='-9e -T0' tar -C testfiles -cJf testfiles/iq-to-audio-fixtures.tar.xz \
  fc-132334577Hz-ft-132300000-AM.wav \
  fc-456834049Hz-ft-455837500-ft2-456872500-NFM.wav
```

Then:
1. Upload the new archive to Google Drive (replace the existing file)
2. Calculate the new SHA256 checksum
3. Update the `GDRIVE_FILE_SHA256` secret in GitHub repository settings

## How It Works

1. **GitHub Actions**: The composite action at `.github/actions/download-test-fixtures`
   uses rclone to download the archive from Google Drive before running tests

2. **Local Development**: The pytest session fixture in `tests/conftest.py` attempts
   to download the archive if it's missing and credentials are available

3. **Extraction**: Test fixtures automatically extract individual WAV files from the
   archive on-demand (see `_extract_fixture()` in `tests/conftest.py`)

The raw `.wav` files remain ignored by git and can be deleted safely after use—they'll
be re-extracted automatically when tests run.
