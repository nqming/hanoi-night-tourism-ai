# Multi-layer production Dockerfile for Hanoi Night-Tourism AI Platform
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install minimal system dependencies for healthcheck & numpy/scipy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (keeps image compact ~700MB instead of ~5GB CUDA)
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

# Install application dependencies
COPY requirements.txt .
RUN grep -v "^torch" requirements.txt > requirements.notorch.txt && \
    pip install -r requirements.notorch.txt && \
    rm requirements.notorch.txt

# Copy source code, crawled Hanoi POI dataset, and trained Contrastive Learning checkpoints
COPY data/ ./data/
COPY ml/ ./ml/
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY tests/ ./tests/

# Create non-root user for container security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
