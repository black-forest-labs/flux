# Specify multi-platform base images
FROM --platform=$TARGETPLATFORM ubuntu:22.04 as base

# Set architecture-specific variables
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Set up Python and basic dependencies
FROM base as python-deps
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies based on architecture
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create venv and install dependencies
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory to parent of docker directory
WORKDIR /build

# Copy the entire project
COPY .. .

# Install PyTorch based on architecture
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        # Install PyTorch for Apple Silicon
        pip install --no-cache-dir torch torchvision torchaudio; \
    else \
        # Install CUDA version for NVIDIA GPUs
        pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118; \
    fi

# Install the package with all dependencies
RUN pip install -e ".[all]"

# Final stage
FROM base
COPY --from=python-deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
# Copy the installed package and its dependencies
COPY --from=python-deps /build /app

# Create volume mount points
VOLUME ["/app/outputs", "/app/models"]

# Set environment variables
ENV TORCH_HOME=/app/models
ENV HF_HOME=/app/models

# Default command that can be overridden
ENTRYPOINT ["python3", "-m", "flux.api"]
