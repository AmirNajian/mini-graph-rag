"""
Data models for the knowledge graph.
"""

from pydantic import BaseModel

class Relation(BaseModel):
    """Represents a relation between entities."""
    source_id: str
    target_id: str
    weight: float = 1.0
    edge_type: str = "cooccurrence"
