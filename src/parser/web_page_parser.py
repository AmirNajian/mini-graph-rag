"""
Utilities for turning web pages into `/ingest`-ready documents.

The core idea is:

- Fetch HTML from one or more URLs.
- Strip HTML tags (ignoring scripts/styles).
- Normalize whitespace into readable paragraphs.
- Return a list of ``{"id": ..., "text": ...}`` dictionaries that matches
  what `/ingest` in ``src/api/main.py`` expects.

Example:

    from src.parser.web_page_parser import build_documents_from_urls
    from src.script.persist_workflow import post_json

    BASE_URL = "http://localhost:8000"
    urls = ["https://www.ecfr.gov/current/title-40"]

    documents = build_documents_from_urls(urls)
    payload = {"documents": documents}
    resp = post_json(f"{BASE_URL}/ingest", payload)
    print(resp)

"""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Dict, Iterable, List, Sequence
from urllib.parse import urlparse
import re
import urllib.request

from src.schema.parser import WebPageDocument


class _TextExtractor(HTMLParser):
    """HTML parser that extracts visible text and drops scripts/styles."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._texts: List[str] = []
        self._ignore_depth: int = 0  # track nested script/style/noscript

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag.lower() in {"script", "style", "noscript"}:
            self._ignore_depth += 1

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag.lower() in {"script", "style", "noscript"} and self._ignore_depth > 0:
            self._ignore_depth -= 1

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._ignore_depth > 0:
            return
        text = data.strip()
        if text:
            self._texts.append(text)

    def get_text(self) -> str:
        """Return the concatenated extracted text."""
        return "\n".join(self._texts)


def fetch_html(url: str, timeout: float = 20.0) -> str:
    """
    Fetch raw HTML from a URL.

    Uses ``urllib.request`` (no external dependencies). Any networking
    errors are propagated to the caller as ``URLError`` / ``HTTPError``.
    """
    req = urllib.request.Request(
        url,
        headers={
            # A simple browser-y UA string; some sites are picky.
            "User-Agent": "mini-graph-rag-web-parser/0.1 (+https://github.com/)",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        # Try to respect the server's declared encoding; default to utf-8.
        content_type = resp.headers.get("Content-Type", "")
        charset = "utf-8"
        match = re.search(r"charset=([^\s;]+)", content_type, re.IGNORECASE)
        if match:
            charset = match.group(1).strip()
        raw = resp.read()
        return raw.decode(charset, errors="replace")


def html_to_text(html: str) -> str:
    """
    Convert an HTML document to a cleaned plaintext string.

    - Drops script/style/noscript content.
    - Collapses excessive whitespace.
    - Preserves basic paragraph breaks.
    """
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    text = parser.get_text()

    # Normalize whitespace: collapse runs of spaces/tabs, tidy newlines.
    # First, normalize Windows/Mac newlines to ``\\n``.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse spaces within lines.
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ newlines to 2 (paragraph breaks).
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing spaces on each line.
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    return text.strip()


def url_to_document_id(url: str) -> str:
    """
    Build a reasonably stable document ID from a URL.

    For example:

    - ``https://www.ecfr.gov/current/title-40`` ->
      ``www.ecfr.gov_current_title-40``
    """
    parsed = urlparse(url)
    # Use path and fragment to discriminate variants if needed.
    path = parsed.path.rstrip("/") or "/"
    fragment = f"#{parsed.fragment}" if parsed.fragment else ""
    raw = f"{parsed.netloc}{path}{fragment}"
    # Replace separators with underscores; keep other characters as-is.
    doc_id = re.sub(r"[^\w\-]+", "_", raw).strip("_")
    return doc_id or "web_page"


def html_to_document(html: str, *, id_hint: str | None = None) -> WebPageDocument:
    """
    Turn HTML into a single ``WebPageDocument``.

    Args:
        html: Raw HTML source.
        id_hint: Optional ID for the document (e.g. a URL or slug). If not
            provided, a generic ``web_page`` ID is used.
    """
    text = html_to_text(html)
    doc_id = id_hint if id_hint is not None else "web_page"
    return WebPageDocument(id=doc_id, text=text)


def build_document_from_url(url: str, *, timeout: float = 20.0) -> WebPageDocument:
    """
    Fetch a single URL and return it as a ``WebPageDocument``.

    This is a convenience wrapper around ``fetch_html`` and
    ``html_to_document`` that also derives a stable document ID
    from the URL.
    """
    html = fetch_html(url, timeout=timeout)
    doc_id = url_to_document_id(url)
    return html_to_document(html, id_hint=doc_id)


def build_documents_from_urls(
    urls: Sequence[str], *, timeout: float = 20.0
) -> List[Dict[str, str]]:
    """
    Fetch multiple URLs and convert them into `/ingest`-ready dicts.

    Args:
        urls: Iterable of URL strings to fetch.
        timeout: Per-request timeout in seconds.

    Returns:
        List of ``{"id": ..., "text": ...}`` dictionaries ready to send
        as the ``documents`` field of an ``IngestRequest``.

    Example:

        from src.parser.web_page_parser import build_documents_from_urls
        docs = build_documents_from_urls(
            ["https://www.ecfr.gov/current/title-40"]
        )
        payload = {"documents": docs}
        # POST payload to /ingest
    """
    documents: List[Dict[str, str]] = []
    for url in urls:
        doc = build_document_from_url(url, timeout=timeout)
        documents.append(doc.to_ingest_dict())
    return documents


def build_documents_from_html_snippets(
    html_snippets: Iterable[str], *, id_prefix: str = "snippet"
) -> List[Dict[str, str]]:
    """
    Convert pre-fetched HTML snippets into `/ingest`-ready documents.

    This is useful if you obtain content via another API (for example,
    the official eCFR API) and just need to normalize it for ingestion.

    Args:
        html_snippets: Iterable of HTML strings.
        id_prefix: Prefix used when generating synthetic IDs.
    """
    documents: List[Dict[str, str]] = []
    for idx, html in enumerate(html_snippets, start=1):
        doc = html_to_document(html, id_hint=f"{id_prefix}_{idx}")
        documents.append(doc.to_ingest_dict())
    return documents


__all__ = [
    "fetch_html",
    "html_to_text",
    "html_to_document",
    "url_to_document_id",
    "build_document_from_url",
    "build_documents_from_urls",
    "build_documents_from_html_snippets",
]
