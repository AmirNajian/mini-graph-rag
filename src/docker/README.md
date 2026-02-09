# Docker

Run Mini GraphRAG in Linux containers so the `lancedb` dependency (from `graphrag`) installs via manylinux wheels and avoids the macOS x86_64 wheel error.

## Data persistence

Graph and LanceDB data are stored in a **local bind-mounted directory** so they survive container restarts and are visible on your machine:

- **Host path:** `./data` at the repo root (created automatically when the app first writes data).
- **In the app container:** `/app/data` (and `DATA_DIR=/app/data`).
- **In the lancedb container:** `/data`.

Both services use the same `./data` directory. The `data/` folder is listed in `.gitignore` so persisted content is not committed.

## Usage

**From the repo root:**

```bash
# Build and run API + lancedb data container
docker compose -f src/docker/docker-compose.yaml up --build

# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

**Run the example script in a one-off container:**

```bash
docker compose -f src/docker/docker-compose.yaml run --rm app uv run python src/script/example.py
```

**Optional:** Create the data directory before first run so you can inspect it:

```bash
mkdir -p data
docker compose -f src/docker/docker-compose.yaml up --build
```

## Services

- **app:** FastAPI server (port 8000). Mounts `./data` at `/app/data` for graph/LanceDB persistence.
- **lancedb:** Data container that holds the same `./data` mount; use it if you add LanceDB or GraphRAG file-based storage later.
