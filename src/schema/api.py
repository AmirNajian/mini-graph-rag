"""
Data models for Mini GraphRAG API.
"""
from typing import List
from pydantic import BaseModel

from .entity import Entity
from .graph import Graph
from .retrieval import RetrievalResult

class DocumentInput(BaseModel):
    """Input document model."""
    id: str
    text: str


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    documents: List[DocumentInput]


class AnswerRequest(BaseModel):
    """Request model for query answering."""
    query: str
    top_k: int = 5


class Citation(BaseModel):
    """Citation model."""
    doc_id: str
    fragment: str
    start_char: int
    end_char: int


class GraphTrace(BaseModel):
    """Graph trace model."""
    query_entities: List[str]
    expanded_entities: List[str]
    top_docs: List[str]
    reasoning: str


class AnswerResponse(BaseModel):
    """Response model for answer endpoint."""
    answer: str
    citations: List[Citation]
    graph_trace: GraphTrace


class IngestResponse(BaseModel):
    """Response model for ingest endpoint."""
    status: str
    documents_processed: int
    entities_extracted: int
    entities_resolved: int
    graph_nodes: int
    graph_edges: int