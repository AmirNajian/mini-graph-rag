# Docker

Run Mini GraphRAG in Linux containers so the `lancedb` dependency (from `graphrag`) installs via manylinux wheels and avoids the macOS x86_64 wheel error.

**From the repo root:**

```bash
# Build and run API + lancedb data container
docker compose -f src/docker/docker-compose.yaml up --build

# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

**Run the example script in a one-off container:**

```bash
docker compose -f src/docker/docker-compose.yaml run --rm app .venv/bin/python src/script/example.py
```

The `app` service runs the FastAPI server. The `lancedb` service is a data container that shares the `lancedb-data` volume with the app for optional persistence.
