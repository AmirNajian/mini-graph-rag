"""
FastAPI application for Mini GraphRAG service.
"""
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import Config
from ingest_pipeline import IngestPipeline
from retriever import Retriever
from schema import DocumentInput, IngestRequest, IngestResponse, AnswerRequest, AnswerResponse, Citation, GraphTrace
from synthesizer import Synthesizer

app = FastAPI(title="Mini GraphRAG", version="0.1.0")

# Global state
config: Optional[Config] = None
ingest_pipeline: Optional[IngestPipeline] = None
retriever: Optional[Retriever] = None
synthesizer: Optional[Synthesizer] = None



@app.on_event("startup")
async def startup_event():
    """Initialize global state on startup."""
    global config, ingest_pipeline, retriever, synthesizer
    
    config = Config()
    ingest_pipeline = IngestPipeline(config)
    retriever = Retriever(config)
    synthesizer = Synthesizer(config)


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
    if ingest_pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    documents = [{"id": doc.id, "text": doc.text} for doc in request.documents]
    stats = ingest_pipeline.run(documents)
    
    # Update retriever with new index
    if retriever is not None:
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
    if retriever is None or synthesizer is None:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    if ingest_pipeline is None or ingest_pipeline.knowledge_graph is None:
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
    )
    
    return AnswerResponse(
        answer=result["answer"],
        citations=result["citations"],
        graph_trace=result["graph_trace"],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

