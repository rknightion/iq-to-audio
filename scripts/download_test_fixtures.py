#!/usr/bin/env python3
"""
Download test fixtures from Google Drive using rclone.

This script downloads the iq-to-audio-fixtures.tar.xz file from Google Drive.
It can be used locally or in CI/CD environments.

Environment variables:
    GDRIVE_SERVICE_ACCOUNT_JSON: Service account JSON credentials
    GDRIVE_FILE_ID: Google Drive file ID
    GDRIVE_FILE_SHA256: Expected SHA256 checksum
    GDRIVE_CLIENT_ID: Optional custom OAuth client ID
    GDRIVE_CLIENT_SECRET: Optional custom OAuth client secret
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LOG = logging.getLogger(__name__)


def check_rclone_installed() -> bool:
    """Check if rclone is installed and available."""
    return shutil.which("rclone") is not None


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with file_path.open("rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def setup_rclone_config(
    service_account_json: str,
    client_id: str | None = None,
    client_secret: str | None = None,
) -> tuple[Path, Path]:
    """
    Set up rclone configuration for Google Drive.

    Returns:
        Tuple of (config_file_path, service_account_file_path)
    """
    # Create temp directory for rclone config
    config_dir = Path.home() / ".config" / "rclone"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "rclone.conf"

    # Write service account JSON to temp file
    service_account_file = Path(tempfile.gettempdir()) / "gdrive-service-account.json"
    service_account_file.write_text(service_account_json)

    # Build rclone config
    config_content = f"""[gdrive]
type = drive
scope = drive.readonly
service_account_file = {service_account_file}
"""

    if client_id and client_secret:
        config_content += f"""client_id = {client_id}
client_secret = {client_secret}
"""
        LOG.info("Using custom OAuth client to avoid rate limits")
    else:
        LOG.info("Using rclone's default client (may be rate limited)")

    config_file.write_text(config_content)
    LOG.info("rclone configuration created at %s", config_file)

    return config_file, service_account_file


def download_fixture(
    file_id: str,
    destination: Path,
    expected_sha256: str | None = None,
) -> bool:
    """
    Download test fixture from Google Drive using rclone.

    Args:
        file_id: Google Drive file ID
        destination: Destination file path
        expected_sha256: Expected SHA256 checksum for validation

    Returns:
        True if download and validation successful, False otherwise
    """
    if not check_rclone_installed():
        LOG.error("rclone is not installed. Please install it first.")
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)

    # Download using rclone
    LOG.info("Downloading test fixtures from Google Drive (file_id: %s)", file_id)

    try:
        result = subprocess.run(
            [
                "rclone",
                "copy",
                "--drive-acknowledge-abuse",
                "--progress",
                "--transfers",
                "1",
                f"gdrive:{file_id}",
                str(destination.parent),
                "-v",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        LOG.info("rclone output: %s", result.stdout)
    except subprocess.CalledProcessError as e:
        LOG.error("rclone failed: %s", e.stderr)
        return False

    # Find the downloaded file (rclone uses the Drive filename)
    downloaded_files = list(destination.parent.glob("*.tar.xz"))
    if not downloaded_files:
        LOG.error("Downloaded file not found in %s", destination.parent)
        return False

    downloaded_file = downloaded_files[0]

    # Rename to expected name if needed
    if downloaded_file != destination:
        downloaded_file.rename(destination)
        LOG.info("Renamed %s to %s", downloaded_file, destination)

    LOG.info("Download complete: %s (%.2f MB)", destination, destination.stat().st_size / 1024 / 1024)

    # Verify checksum if provided
    if expected_sha256:
        LOG.info("Verifying SHA256 checksum...")
        actual_sha256 = compute_sha256(destination)
        if actual_sha256 != expected_sha256:
            LOG.error("Checksum mismatch!")
            LOG.error("Expected: %s", expected_sha256)
            LOG.error("Got:      %s", actual_sha256)
            return False
        LOG.info("Checksum verified successfully")

    return True


def cleanup(config_file: Path | None = None, service_account_file: Path | None = None):
    """Clean up temporary files."""
    if config_file and config_file.exists():
        config_file.unlink()
        LOG.info("Removed %s", config_file)

    if service_account_file and service_account_file.exists():
        service_account_file.unlink()
        LOG.info("Removed %s", service_account_file)


def main() -> int:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Get environment variables
    service_account_json = os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON")
    file_id = os.getenv("GDRIVE_FILE_ID")
    expected_sha256 = os.getenv("GDRIVE_FILE_SHA256")
    client_id = os.getenv("GDRIVE_CLIENT_ID")
    client_secret = os.getenv("GDRIVE_CLIENT_SECRET")

    # Validate required variables
    if not service_account_json or not file_id:
        LOG.warning(
            "GDRIVE_SERVICE_ACCOUNT_JSON or GDRIVE_FILE_ID not set. "
            "Test fixtures will not be downloaded."
        )
        return 0  # Not an error - gracefully skip

    # Check if file already exists
    repo_root = Path(__file__).parent.parent
    destination = repo_root / "testfiles" / "iq-to-audio-fixtures.tar.xz"

    if destination.exists():
        LOG.info("Test fixtures already present at %s", destination)
        if expected_sha256:
            actual_sha256 = compute_sha256(destination)
            if actual_sha256 == expected_sha256:
                LOG.info("Checksum verified - no download needed")
                return 0
            LOG.warning("Checksum mismatch - re-downloading")

    # Set up rclone config
    config_file = None
    service_account_file = None
    try:
        config_file, service_account_file = setup_rclone_config(
            service_account_json,
            client_id,
            client_secret,
        )

        # Download fixture
        success = download_fixture(file_id, destination, expected_sha256)
        return 0 if success else 1

    except Exception as e:
        LOG.exception("Failed to download test fixtures: %s", e)
        return 1

    finally:
        cleanup(config_file, service_account_file)


if __name__ == "__main__":
    sys.exit(main())
