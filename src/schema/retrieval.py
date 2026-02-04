from typing import List

from pydantic import BaseModel

class RetrievalResult(BaseModel):
    """Result of a retrieval operation."""
    doc_id: str
    score: float
    lexical_score: float
    graph_score: float
    matched_entities: List[str]
