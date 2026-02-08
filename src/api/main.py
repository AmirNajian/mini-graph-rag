"""
FastAPI application for Mini GraphRAG service.
"""
from fastapi import FastAPI, HTTPException

from src.config.config import Config
from src.ingest.ingest_pipeline import IngestPipeline
from src.retrieval.retriever import Retriever
from src.schema.api import IngestRequest, IngestResponse, AnswerRequest, AnswerResponse, Citation, GraphTrace
from src.retrieval.synthesizer import Synthesizer

app = FastAPI(title="Mini GraphRAG", version="0.1.0")

# Global state — initialized at import so they're never None when handling requests
_config = Config()
config: Config = _config
ingest_pipeline: IngestPipeline = IngestPipeline(_config)
retriever: Retriever = Retriever(_config)
synthesizer: Synthesizer = Synthesizer(_config)


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

