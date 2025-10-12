# syntax=docker/dockerfile:1.7
# Multi-stage Dockerfile for iq-to-audio
# Supports both CLI and GUI modes (GUI requires X11 forwarding)

# Stage 1: Base image with system dependencies
FROM python:3.13-slim AS base

# Reduce apt noise and keep image small
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies including FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    libxcb-cursor0 \
    libegl1 \
    libxcb1 \
    libxrender1 \
    libxi6 \
    libsm6 \
    libxext6 \
    libxfixes3 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxrandr2 \
    libxtst6 \
    libxss1 \
    libasound2 \
    libpulse0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
ENV UV_INSTALL_DIR=/usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && uv --version

# Stage 2: Build stage with dev dependencies
FROM base AS builder

WORKDIR /app

# Copy dependency manifests first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Copy source files needed for package build
COPY README.md README.md
COPY src/ src/
COPY packaging/ packaging/
COPY tools/ tools/

# Install Python dependencies using uv (respect the lock file)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Generate icons (if GUI is needed)
RUN uv run python tools/generate_app_icons.py || true

# Stage 3: Runtime stage
FROM base AS runtime

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code
COPY --from=builder /app/src /app/src
COPY --from=builder /app/packaging /app/packaging
COPY --from=builder /app/README.md /app/README.md
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# For GUI support
ENV QT_DEBUG_PLUGINS=0
ENV QT_X11_NO_MITSHM=1

# Create non-root user for security
RUN useradd -m -s /bin/bash iquser && \
    chown -R iquser:iquser /app

USER iquser

# Health check for CLI
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "from iq_to_audio import cli; print('OK')" || exit 1

# Default to CLI help
ENTRYPOINT ["python", "-m", "iq_to_audio.cli"]
CMD ["--help"]

# Stage 4: Development stage (optional, includes dev tools)
FROM builder AS development

# Install development dependencies
RUN uv sync --frozen --dev

# Install additional dev tools
RUN apt-get update && apt-get install -y \
    vim \
    git \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Keep the container running for development
CMD ["/bin/bash"]

# Usage instructions:
#
# Build for production (CLI):
#   docker build -t iq-to-audio:latest .
#
# Build for development:
#   docker build --target development -t iq-to-audio:dev .
#
# Run CLI mode:
#   docker run --rm -v $(pwd)/data:/data iq-to-audio:latest \
#     --interactive /data/recording.wav
#
# Run GUI mode with X11 forwarding (Linux):
#   docker run --rm \
#     -e DISPLAY=$DISPLAY \
#     -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
#     -v $(pwd)/data:/data \
#     iq-to-audio:latest --interactive
#
# Run GUI mode (macOS with XQuartz):
#   xhost + 127.0.0.1
#   docker run --rm \
#     -e DISPLAY=host.docker.internal:0 \
#     -v $(pwd)/data:/data \
#     iq-to-audio:latest --interactive
#
# Run GUI mode (Windows with VcXsrv):
#   docker run --rm \
#     -e DISPLAY=host.docker.internal:0.0 \
#     -v %cd%/data:/data \
#     iq-to-audio:latest --interactive
