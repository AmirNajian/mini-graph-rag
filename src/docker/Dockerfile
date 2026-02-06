# Use Linux base so lancedb (transitive via graphrag) has manylinux wheels.
# On macOS x86_64 with newer SDK (macosx_26_0) no wheel exists; Linux avoids that.
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_SYSTEM_PYTHON=1

# Dependencies first for layer cache (no project install; we use PYTHONPATH)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project

# Project code
COPY src ./src

# Run from project root; PYTHONPATH so "config", "ingest_pipeline", etc. resolve from src/
ENV PYTHONPATH=/app/src
EXPOSE 8000

# Default: run the FastAPI app
CMD [".venv/bin/python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
