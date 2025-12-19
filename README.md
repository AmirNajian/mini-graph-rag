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
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

### API Endpoints

#### `POST /ingest`

Ingest documents into the knowledge graph.

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

