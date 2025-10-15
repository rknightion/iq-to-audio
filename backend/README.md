# Backend Utilities Docker Image

Multi-architecture Docker image containing third-party SDR utilities built from upstream sources.

## Contents

This image includes:
- **dsd-fme** (Digital Speech Decoder: Florida Man Edition) - Digital voice decoder
- **mbelib** - Multi-Band Excitation voice codec library

## Quick Start

### Pull the Image

```bash
docker pull ghcr.io/rknightion/iq-to-audio-backend:latest
```

### Run dsd-fme

```bash
# Show help
docker run --rm ghcr.io/rknightion/iq-to-audio-backend:latest --help

# Process audio file
docker run --rm \
  -v $(pwd)/data:/data \
  ghcr.io/rknightion/iq-to-audio-backend:latest \
  -i /data/input.bin -o /data/output.wav
```

## Available Tags

| Tag | Description | Update Frequency |
|-----|-------------|-----------------|
| `latest` | Latest stable build from main branch | On push to main |
| `nightly` | Nightly automated build | Daily at 2 AM UTC |
| `main` | Latest main branch build | On push to main |
| `main-<sha>` | Specific commit SHA | Per commit |
| `YYYYMMDD` | Date-based nightly tags | Daily |
| `pr-<number>` | Pull request preview | Per PR update |

## Supported Platforms

- `linux/amd64` (x86_64)
- `linux/arm64` (aarch64)

Both platforms are built and tested automatically in CI.

## Upstream Sources

This image builds from upstream repositories without maintaining a fork:

| Component | Repository | Branch |
|-----------|-----------|--------|
| mbelib | [lwvmobile/mbelib](https://github.com/lwvmobile/mbelib) | `ambe_tones` |
| dsd-fme | [lwvmobile/dsd-fme](https://github.com/lwvmobile/dsd-fme) | `audio_work` |

### Custom Branch Builds

You can trigger builds with different upstream branches via GitHub Actions:

1. Go to [Actions → Build Backend Utils Image](https://github.com/rknightion/iq-to-audio/actions/workflows/backend-docker.yml)
2. Click "Run workflow"
3. Specify custom branches for mbelib and/or dsd-fme
4. The image will be tagged with the commit SHA

## Build Process

### Multi-Stage Build Strategy

The Dockerfile uses a three-stage build to minimize image size:

```
Stage 1: mbelib-builder
  ├─ Clone mbelib repository
  ├─ Build with CMake
  └─ Install to /usr/local

Stage 2: app-builder
  ├─ Copy mbelib artifacts from Stage 1
  ├─ Clone dsd-fme repository
  ├─ Build with CMake
  └─ Install to /usr/local

Stage 3: runtime
  ├─ Copy only binaries and libraries
  ├─ Install runtime dependencies (no build tools)
  ├─ Run as non-root user (ubuntu, UID 1000)
  └─ Set entrypoint to dsd-fme
```

### Local Build

```bash
# Build for your platform
docker build -t backend:local ./backend

# Build multiarch (requires buildx)
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t backend:multiarch \
  ./backend
```

## CI/CD Pipeline

### Automatic Builds

The image is automatically built and published in these scenarios:

1. **Push to main**: Triggers build with `latest`, `main`, `nightly`, and `main-<sha>` tags
2. **Pull requests**: Builds but doesn't push, runs smoke tests
3. **Nightly schedule**: Daily build at 2 AM UTC with date tag
4. **Manual trigger**: On-demand builds via GitHub Actions UI

### Security Features

Every build includes:

✅ **Multi-architecture support** (amd64, arm64)
✅ **Trivy vulnerability scanning** (CRITICAL and HIGH severity)
✅ **SBOM generation** (SPDX format)
✅ **Non-root runtime** (ubuntu user, UID 1000)
✅ **Layer caching** (GitHub Actions cache)
✅ **Minimal attack surface** (only runtime dependencies)

### Testing

Pull requests run these smoke tests:

```bash
# Binary functionality test
docker run --rm <image> --help

# Non-root verification
docker run --rm <image> sh -c 'id -u' | grep -q "1000"
```

Main branch builds additionally verify:

```bash
# Multiarch manifest inspection
docker buildx imagetools inspect <image>:latest
```

## Security

### Vulnerability Scanning

All published images are scanned with Trivy. Results are uploaded to the [Security tab](https://github.com/rknightion/iq-to-audio/security/code-scanning).

To scan locally:

```bash
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest image \
  ghcr.io/rknightion/iq-to-audio-backend:latest
```

### SBOM (Software Bill of Materials)

Each build generates an SBOM in SPDX format, available as a workflow artifact for 90 days.

### Non-Root User

The image runs as the `ubuntu` user (UID 1000, GID 1000) for security. If you need to access files in mounted volumes, ensure permissions allow UID 1000:

```bash
# Set ownership on host
sudo chown -R 1000:1000 ./data

# Or use Docker's user flag
docker run --user $(id -u):$(id -g) ...
```

## Troubleshooting

### Permission Denied on Mounted Volumes

```bash
# Option 1: Change host directory ownership
sudo chown -R 1000:1000 ./data

# Option 2: Run as your user (less secure)
docker run --user $(id -u):$(id -g) \
  ghcr.io/rknightion/iq-to-audio-backend:latest ...
```

### Platform Mismatch

```bash
# Force specific platform
docker run --platform linux/amd64 \
  ghcr.io/rknightion/iq-to-audio-backend:latest ...
```

### View Image Details

```bash
# Inspect multiarch manifest
docker manifest inspect ghcr.io/rknightion/iq-to-audio-backend:latest

# Check available platforms
docker buildx imagetools inspect ghcr.io/rknightion/iq-to-audio-backend:latest
```

## Development

### Modifying the Dockerfile

1. Edit `backend/Dockerfile`
2. Test locally:
   ```bash
   docker build -t backend:test ./backend
   docker run --rm backend:test --help
   ```
3. Create PR - CI will build and test automatically
4. Merge to main - image will be published

### Path Triggers

The workflow only runs when these paths change:
- `backend/**` (any file in backend directory)
- `.github/workflows/backend-docker.yml` (workflow file itself)

### Workflow Configuration

Key workflow settings in `.github/workflows/backend-docker.yml`:

```yaml
env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}-backend

jobs:
  build-and-push:
    permissions:
      contents: read        # Read repository
      packages: write       # Push to GHCR
      security-events: write # Upload Trivy results
```

## Resources

- [Dockerfile](./Dockerfile)
- [GitHub Actions Workflow](../.github/workflows/backend-docker.yml)
- [dsd-fme Documentation](https://github.com/lwvmobile/dsd-fme)
- [mbelib Repository](https://github.com/lwvmobile/mbelib)
- [Container Registry](https://github.com/rknightion/iq-to-audio/pkgs/container/iq-to-audio-backend)

## License

This Docker image packages upstream utilities with their respective licenses:
- dsd-fme: ISC License
- mbelib: ISC License

See upstream repositories for detailed license information.
