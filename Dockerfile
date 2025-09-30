# VibeVoice Inference Server Dockerfile

# Build stage
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 as builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create and activate virtual environment
RUN python3.10 -m pip install --upgrade pip setuptools wheel
RUN python3.10 -m pip install virtualenv
RUN python3.10 -m virtualenv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better Docker layer caching
COPY requirements.txt /tmp/requirements.txt

# Install build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    pkg-config && \
    \
    # Install Python dependencies from requirements.txt
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    \
    # Attempt to install flash-attn (optional, continue if fails)
    [cite_start]pip install --no-cache-dir flash-attn --no-build-isolation || echo "flash-attn installation failed, continuing without it" [cite: 2, 3] && \
    \
    # Clean up build-time dependencies and apt caches to save space
    apt-get purge -y --auto-remove build-essential cmake git wget curl libffi-dev libssl-dev libsox-dev pkg-config && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Runtime stage
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"
ENV CUDA_VISIBLE_DEVICES=0

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    libsndfile1 \
    ffmpeg \
    sox \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create app user for security
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy application code
COPY --chown=app:app vibevoice/ ./vibevoice/
COPY --chown=app:app speak.py ./ 
COPY --chown=app:app requirements.txt ./ 

# Create models directory with proper permissions
RUN mkdir -p ./models

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1 

# Expose port
EXPOSE 8000

# Set default command
CMD ["python", "-m", "uvicorn", "speak:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Labels for metadata
LABEL maintainer="Spruce Emmanuel <hello@spruceemmanuel.com>"
LABEL description="VibeVoice Inference Server with FastAPI"
LABEL version="1.0.0"
LABEL gpu.required="true"