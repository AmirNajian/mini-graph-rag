"""
FastAPI application for Mini GraphRAG service.

This module exposes:
- `/ingest` for ingesting documents into the in-memory graph/index
- `/answer` for answering questions using the current graph/index
- `/graph/export` for exporting the graph to GraphRAG-compatible JSONL files
- `/state/save` and `/state/load` for persisting and restoring the in-memory state

State is stored under the directory given by the `DATA_DIR` environment
variable (default: `data/`), which is bind-mounted in Docker.
"""
import os
import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.config.config import Config
from src.ingest.ingest_pipeline import IngestPipeline
from src.retrieval.retriever import Retriever
from src.schema.api import (
    IngestRequest,
    IngestResponse,
    AnswerRequest,
    AnswerResponse,
    Citation,
    GraphTrace,
)
from src.retrieval.synthesizer import Synthesizer

app = FastAPI(title="Mini GraphRAG", version="0.1.0")


def _get_data_dir() -> Path:
    """Return the base data directory for persistence (and ensure it exists)."""
    base = os.getenv("DATA_DIR", "data")
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


# Global state — initialized at import so they're never None when handling requests
_config = Config()
config: Config = _config
ingest_pipeline: IngestPipeline = IngestPipeline(_config)
retriever: Retriever = Retriever(_config)
synthesizer: Synthesizer = Synthesizer(_config)


def _save_state() -> Path:
    """
    Persist the current in-memory state (config, ingest pipeline, retriever).

    This is a simple pickle-based snapshot that can be restored later using
    `_load_state`. The returned path points to the created/overwritten file.
    """
    data_dir = _get_data_dir()
    state_path = data_dir / "state.pkl"

    # Note: this is trusted, local persistence only (pickle is not safe for
    # untrusted inputs).
    with state_path.open("wb") as f:
        pickle.dump(
            {
                "config": config,
                "ingest_pipeline": ingest_pipeline,
                "retriever": retriever,
            },
            f,
        )

    return state_path


def _load_state() -> Path:
    """
    Load previously persisted state from disk and update globals.

    Returns:
        Path to the state file that was loaded.

    Raises:
        FileNotFoundError: If no saved state exists yet.
        pickle.PickleError: If the state file is corrupted or incompatible.
    """
    global config, ingest_pipeline, retriever, synthesizer

    data_dir = _get_data_dir()
    state_path = data_dir / "state.pkl"
    if not state_path.exists():
        raise FileNotFoundError(state_path)

    with state_path.open("rb") as f:
        state = pickle.load(f)

    config = state["config"]
    ingest_pipeline = state["ingest_pipeline"]
    retriever = state["retriever"]
    # Recreate synthesizer from config; its state is derived from inputs.
    synthesizer = Synthesizer(config)

    return state_path


def _export_graph(subdir: str = "graph") -> Path:
    """
    Export the current knowledge graph to GraphRAG-compatible JSONL files.

    Args:
        subdir: Subdirectory under DATA_DIR to write into (default: ``graph``).

    Returns:
        The path where graph files were written.
    """
    if ingest_pipeline is None or ingest_pipeline.knowledge_graph is None:
        raise HTTPException(
            status_code=400,
            detail="No graph available to export. Ingest documents first.",
        )

    data_dir = _get_data_dir()
    output_path = data_dir / subdir
    ingest_pipeline.knowledge_graph.export_to_graphrag_format(output_path)
    return output_path


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mini-graph-rag"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents into the knowledge graph.
    
    Args:
        request: List of documents to ingest
        
    Returns:
        IngestResponse with statistics
    """
    documents = [{"id": doc.id, "text": doc.text} for doc in request.documents]
    stats = ingest_pipeline.run(documents)

    retriever.update_index(
        ingest_pipeline.text_indexer,
        ingest_pipeline.knowledge_graph,
        ingest_pipeline.entity_resolver,
    )

    # Persist state and export graph after each ingestion so that the
    # in-memory graph/index can be restored and inspected later.
    _save_state()
    _export_graph(subdir="graph")

    return IngestResponse(
        status="success",
        documents_processed=stats["documents_processed"],
        entities_extracted=stats["entities_extracted"],
        entities_resolved=stats["entities_resolved"],
        graph_nodes=stats["graph_nodes"],
        graph_edges=stats["graph_edges"],
    )


@app.post("/answer", response_model=AnswerResponse)
async def answer_query(request: AnswerRequest):
    """
    Answer a query using the knowledge graph.
    
    Args:
        request: Query string and parameters
        
    Returns:
        AnswerResponse with answer, citations, and graph trace
    """
    # Retriever is only populated after /ingest calls update_index()
    if retriever.text_indexer is None or retriever.knowledge_graph is None:
        raise HTTPException(
            status_code=400, detail="No documents ingested. Call /ingest first."
        )
    
    # Retrieve relevant documents
    retrieval_results = retriever.retrieve(request.query, top_k=request.top_k)
    
    # Synthesize answer
    result = synthesizer.generate(
        retrieval_results,
        request.query,
        ingest_pipeline.knowledge_graph,
        ingest_pipeline.text_indexer,
    )

    return AnswerResponse(
        answer=result["answer"],
        citations=result["citations"],
        graph_trace=result["graph_trace"],
    )


@app.post("/graph/export")
async def export_graph(subdir: str = "graph") -> dict:
    """
    Export the current knowledge graph to GraphRAG-compatible JSONL files.

    Args:
        subdir: Optional subdirectory name under ``DATA_DIR`` (default: ``graph``).

    Returns:
        JSON with the output path where files were written.
    """
    path = _export_graph(subdir=subdir)
    return {"status": "success", "output_path": str(path)}


@app.post("/state/save")
async def save_state() -> dict:
    """
    Persist the current in-memory state (config, ingest pipeline, retriever).

    This serializes the state into a pickle file under ``DATA_DIR`` so that it
    can be restored later (e.g., after a restart) via ``/state/load``.
    """
    path = _save_state()
    return {"status": "success", "state_path": str(path)}


@app.post("/state/load")
async def load_state() -> dict:
    """
    Restore previously saved in-memory state from disk.

    This reloads the configuration, ingest pipeline, and retriever, and
    recreates the synthesizer from the loaded config.
    """
    try:
        path = _load_state()
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No saved state found. Ingest documents and call /state/save first.",
        )
    except pickle.PickleError as exc:  # type: ignore[attr-defined]
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load saved state: {exc}",
        ) from exc

    return {"status": "success", "state_path": str(path)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

