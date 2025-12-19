"""
Data models for entities.
"""

from pydantic import BaseModel


class Entity(BaseModel):
    """Represents an extracted entity."""
    text: str
    start_char: int
    end_char: int
    confidence: float
    entity_type: str = "UNKNOWN"


class EntityVariant(BaseModel):
    """Represents a variant of an entity."""
    text: str
    canonical_id: str
    confidence: float
