# VibeVoice Inference Server Dockerfile - Optimized
# Build stage
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 as builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Python and build dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libffi-dev \
    libssl-dev \
    libsndfile1-dev \
    ffmpeg \
    sox \
    libsox-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip in venv
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy and install requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Attempt flash-attn installation (optional)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "flash-attn installation failed, continuing without it"

# Clean up build artifacts and unnecessary files
RUN find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -type f -name "*.pyc" -delete \
    && find /opt/venv -type f -name "*.pyo" -delete \
    && find /opt/venv -type d -name "*.dist-info" -exec rm -rf {}/RECORD {} + 2>/dev/null || true \
    && find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -type d -name "test" -exec rm -rf {} + 2>/dev/null || true

# ============================================
# Runtime stage - MINIMAL IMAGE
# ============================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    CUDA_VISIBLE_DEVICES=0 \
    PIP_NO_CACHE_DIR=1

# Install ONLY runtime dependencies (no dev packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    libsndfile1 \
    ffmpeg \
    sox \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /var/cache/apt/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create app user
RUN useradd --create-home --shell /bin/bash app

# Switch to app user
USER app
WORKDIR /home/app

# Copy application code
COPY --chown=app:app vibevoice/ ./vibevoice/
COPY --chown=app:app speak.py ./
COPY --chown=app:app requirements.txt ./

# Create models directory
RUN mkdir -p ./models

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "speak:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Metadata
LABEL maintainer="Spruce Emmanuel <hello@spruceemmanuel.com>" \
      description="VibeVoice Inference Server with FastAPI" \
      version="1.0.0" \
      gpu.required="true"