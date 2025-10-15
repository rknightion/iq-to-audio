# syntax=docker/dockerfile:1

# Digital Speech Decoder: Florida Man Edition (dsd-fme)
# Multi-stage optimized Docker build

# Build arguments for version control
ARG MBELIB_REPO=https://github.com/lwvmobile/mbelib
ARG MBELIB_BRANCH=ambe_tones
ARG DSDFME_REPO=https://github.com/lwvmobile/dsd-fme
ARG DSDFME_BRANCH=audio_work
ARG UBUNTU_VERSION=24.04

# ============================================================================
# Stage 1: Build mbelib (voice codec library)
# ============================================================================
FROM ubuntu:${UBUNTU_VERSION} AS mbelib-builder

ARG MBELIB_REPO
ARG MBELIB_BRANCH

# Install build dependencies for mbelib
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        cmake \
        make \
        gcc \
        g++ \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Clone and build mbelib
WORKDIR /build
RUN git clone --depth 1 --branch ${MBELIB_BRANCH} ${MBELIB_REPO} mbelib && \
    cd mbelib && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install

# ============================================================================
# Stage 2: Build dsd-fme application
# ============================================================================
FROM ubuntu:${UBUNTU_VERSION} AS app-builder

ARG DSDFME_REPO
ARG DSDFME_BRANCH

# Copy mbelib artifacts from previous stage
COPY --from=mbelib-builder /usr/local/lib/libmbe* /usr/local/lib/
COPY --from=mbelib-builder /usr/local/include/mbe* /usr/local/include/
RUN ldconfig

# Install all build dependencies for dsd-fme
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libpulse-dev libsndfile1-dev libfftw3-dev liblapack-dev socat libusb-1.0-0-dev rtl-sdr ca-certificates librtlsdr-dev libusb-1.0-0-dev cmake git wget make build-essential libitpp-dev libncurses-dev libncurses6 libcodec2-dev \
        socat \
        sox && \
    rm -rf /var/lib/apt/lists/*

# Clone and build dsd-fme
WORKDIR /build
RUN git clone --depth 1 --branch ${DSDFME_BRANCH} ${DSDFME_REPO} dsd-fme && \
    cd dsd-fme && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install

# ============================================================================
# Stage 3: Runtime image (minimal, non-root)
# ============================================================================
FROM ubuntu:${UBUNTU_VERSION} AS runtime

# Metadata labels
LABEL org.opencontainers.image.title="DSD-FME" \
      org.opencontainers.image.description="Digital Speech Decoder: Florida Man Edition" \
      org.opencontainers.image.url="https://github.com/lwvmobile/dsd-fme" \
      org.opencontainers.image.authors="lwvmobile" \
      org.opencontainers.image.licenses="ISC"

# Install only runtime dependencies (no build tools)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        libpulse0 \
        libsndfile1 \
        libfftw3-bin \
        liblapack3 \
        libusb-1.0-0 \
        rtl-sdr \
        librtlsdr2 \
        libitpp8v5 \
        libncurses6 \
        libcodec2-1.2 \
        socat \
        sox && \
    rm -rf /var/lib/apt/lists/*

# Copy compiled libraries and binaries from build stages
COPY --from=mbelib-builder /usr/local/lib/libmbe* /usr/local/lib/
COPY --from=app-builder /usr/local/bin/dsd-fme /usr/local/bin/

# Update library cache
RUN ldconfig

# Create data directory and set ownership for existing ubuntu user
RUN mkdir -p /data && \
    chown -R ubuntu:ubuntu /data

# Switch to non-root user (ubuntu user already exists with UID 1000)
USER ubuntu
WORKDIR /data

# Set entrypoint to dsd-fme binary
ENTRYPOINT ["/usr/local/bin/dsd-fme"]

# Default help command (can be overridden)
CMD ["--help"]
