"""
Data models for the parser.
"""
from typing import Dict

from pydantic import BaseModel


class WebPageDocument(BaseModel):
    """
    Simple representation of a parsed web page.

    This mirrors the structure expected by the ingestion pipeline:
    an identifier and a single text field.
    """

    """The identifier for the document."""
    id: str

    """The text of the document."""
    text: str

    def to_ingest_dict(self) -> Dict[str, str]:
        """
        Return a plain dict suitable for `/ingest`.
        """
        return {"id": self.id, "text": self.text}
