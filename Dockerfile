FROM python:3.11-slim

LABEL maintainer="hermes-rag"
LABEL description="Lightweight, high-precision RAG retrieval optimization framework"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi || \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /app/data/chroma_db /app/logs

# Expose API port
EXPOSE 8000

# Run with optimized settings
CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]