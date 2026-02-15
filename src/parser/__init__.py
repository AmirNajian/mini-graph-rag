"""
Parsing helpers for preparing documents for ingestion.

Currently includes:

- ``web_page_parser``: Utilities for turning raw HTML web pages (or URLs)
  into the ``{"id": ..., "text": ...}`` structures expected by the
  `/ingest` endpoint defined in ``src/api/main.py``.
"""

from .web_page_parser import (  # noqa: F401
    build_document_from_url,
    build_documents_from_html_snippets,
    build_documents_from_urls,
    fetch_html,
    html_to_document,
    html_to_text,
    url_to_document_id,
)

__all__ = [
    "build_document_from_url",
    "build_documents_from_html_snippets",
    "build_documents_from_urls",
    "fetch_html",
    "html_to_document",
    "html_to_text",
    "url_to_document_id",
]

