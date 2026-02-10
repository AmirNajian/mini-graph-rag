# Mini GraphRAG Service

A local, self-contained GraphRAG (Graph Retrieval-Augmented Generation) service that ingests documents, builds a knowledge graph, and answers queries with citations and graph traces.

## Features

- **Entity Extraction**: Heuristic extraction of entities from text (capitalized phrases, acronyms)
- **Entity Resolution**: Normalization and deduplication of entity variants
- **Knowledge Graph**: In-memory graph built using NetworkX with entity co-occurrence relationships
- **Hybrid Retrieval**: Combines TF-IDF lexical similarity with graph-based connectivity
- **Answer Synthesis**: Generates answers with citations and graph traces
- **FastAPI API**: RESTful endpoints for ingestion and querying

## Installation

This project uses `uv` for dependency management. Install dependencies:

```bash
uv sync
```

If you want to use spaCy NER (optional):

```bash
uv sync --extra spacy
python -m spacy download en_core_web_sm
```

## Usage

### Start the API server

```bash
uv run uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`.

### API Endpoints

#### `POST /ingest`

Ingest documents into the knowledge graph and update the in-memory index.
On each call, the service also:

- Saves a serialized snapshot of the current state to `DATA_DIR/state.pkl`
- Exports the knowledge graph to GraphRAG-compatible JSONL files in `DATA_DIR/graph/`

```json
{
  "documents": [
    {
      "id": "doc1",
      "text": "Franklin Templeton is a major investment firm..."
    }
  ]
}
```

#### `POST /answer`

Answer a query using the knowledge graph.

```json
{
  "query": "What is Franklin Templeton?",
  "top_k": 5
}
```

#### `GET /health`

Health check endpoint.

#### `POST /graph/export`

Export the current knowledge graph to GraphRAG-compatible JSONL files. By default,
the data is written to the `graph/` subdirectory under `DATA_DIR` (see below).

#### `POST /state/save`

Persist the current in-memory state (configuration, ingest pipeline, retriever) to
`DATA_DIR/state.pkl`. This is a simple, local-only snapshot used to restore state
after a restart.

#### `POST /state/load`

Load previously saved state from `DATA_DIR/state.pkl` and restore the configuration,
ingest pipeline, and retriever. This allows the service to resume answering queries
without re-ingesting all documents.

### Persistence and `DATA_DIR`

The service uses a base data directory (by default `data/` in the project root)
for:

- `state.pkl`: Pickled snapshot of the graph/index state
- `graph/`: GraphRAG-compatible JSONL export of the knowledge graph

You can override the location via the `DATA_DIR` environment variable.

### Python API

```python
from main import MiniGraphRAG
from config import Config

config = Config.default()
rag = MiniGraphRAG(config)

# Ingest documents
documents = [
    {"id": "doc1", "text": "..."},
    {"id": "doc2", "text": "..."}
]
rag.ingest(documents)

# Answer query
result = rag.answer("What is X?")
print(result.answer)
print(result.citations)
print(result.graph_trace)
```

## Project Structure

- `main.py`: FastAPI application and route handlers
- `entity_extractor.py`: Entity extraction from text
- `entity_resolver.py`: Entity normalization and deduplication
- `knowledge_graph.py`: Knowledge graph implementation (NetworkX)
- `text_indexer.py`: TF-IDF indexing and retrieval
- `retriever.py`: Hybrid retrieval (lexical + graph)
- `synthesizer.py`: Answer synthesis and citation extraction
- `ingest_pipeline.py`: Document ingestion pipeline
- `config.py`: Configuration management

## Configuration

Configuration can be provided via:
1. `config.yaml` file
2. Environment variables
3. Default values

See `config.py` for available configuration options.

## Docker deployment

Run the service in containers (recommended if `lancedb` fails to install on your platform, e.g. macOS x86_64):

```bash
# From repo root
docker compose -f src/docker/docker-compose.yaml up --build
```

Graph and LanceDB data are persisted to a **local directory** on your machine:

- Data is stored in `./data` at the repo root (bind-mounted into the containers).
- It survives restarts and is visible on the host; `data/` is in `.gitignore`.

See [src/docker/README.md](src/docker/README.md) for details, optional commands, and service descriptions.

## Development

Run tests:

```bash
uv run pytest
```

Format code:

```bash
uv run black .
uv run ruff check .
```

## License

MIT

