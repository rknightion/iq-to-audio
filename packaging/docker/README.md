# Bundled Docker Images

This directory contains compressed Docker image tars that are bundled into the PyInstaller distribution for offline-first operation.

## Contents

During the build process, the following files are generated and placed here:

- `backend-amd64.tar.xz` — Backend image for x86_64 (Intel/AMD) systems
- `backend-arm64.tar.xz` — Backend image for ARM64 (Apple Silicon, etc.) systems

## Generation

These tars are created during CI/CD or can be manually generated:

```bash
# Pull and export amd64 image
docker pull --platform linux/amd64 ghcr.io/rknightion/iq-to-audio-backend:latest
docker save ghcr.io/rknightion/iq-to-audio-backend:latest | xz -9 > backend-amd64.tar.xz

# Pull and export arm64 image
docker pull --platform linux/arm64 ghcr.io/rknightion/iq-to-audio-backend:latest
docker save ghcr.io/rknightion/iq-to-audio-backend:latest | xz -9 > backend-arm64.tar.xz
```

## Size

- Uncompressed image: ~162 MB
- xz -9 compressed: ~32 MB per architecture

## Runtime Behavior

The application uses a hybrid approach:

1. On first digital post use, check if the image exists in Docker
2. If missing, load from the bundled tar (this directory)
3. If bundled tar is missing, fall back to `docker pull` (requires internet)

Users can manually update the image via the "Update Container Image" button in the Digital Post page, which pulls the latest version from the registry.

## Build Integration

The PyInstaller spec (`iq-to-audio.spec`) automatically includes the appropriate platform-specific tar based on the build architecture. See `_bundled_docker_image()` function in the spec file.
