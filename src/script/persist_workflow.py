#!/usr/bin/env python3
"""
Example workflow: load data from a JSON file, ingest it, persist state and graph.

This script talks to the Mini GraphRAG API to:
  1. Read documents from a JSON file (default: data/temporal-sdk-data.json).
  2. Ingest them in batches via POST /ingest (which auto-saves state/graph).
  3. Explicitly call POST /state/save and POST /graph/export.
  4. Optionally call POST /state/load to restore after a simulated restart.
  5. Answer a query to verify the graph is usable.

The persisted graph is written under DATA_DIR (e.g. ./data in Docker), which is
the same directory used for LanceDB/graph storage when running in containers:
  - state.pkl: full in-memory state (config, pipeline, retriever).
  - graph/: GraphRAG-compatible JSONL (entities.jsonl, relationships.jsonl).

Run the API first (e.g. docker compose up or uvicorn src.api.main:app), then:

  uv run python src/script/persist_workflow.py
  uv run python src/script/persist_workflow.py --data data/temporal-sdk-data.json
  uv run python src/script/persist_workflow.py --base-url http://localhost:8000
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BATCH_SIZE = 25


def post_json(url: str, data: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_no_body(url: str) -> dict:
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_documents(path: Path) -> list[dict]:
    """Load documents from a JSON file.

    Expected format::

        { "documents": [ {"id": "...", "text": "..."}, ... ] }
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    docs = data.get("documents")
    if not isinstance(docs, list) or not docs:
        print(f"Error: '{path}' must contain a non-empty 'documents' array.", file=sys.stderr)
        sys.exit(1)

    # Validate each entry
    for i, doc in enumerate(docs):
        if "id" not in doc or "text" not in doc:
            print(f"Error: document at index {i} is missing 'id' or 'text'.", file=sys.stderr)
            sys.exit(1)

    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Persist workflow for Mini GraphRAG API")
    parser.add_argument(
        "--data",
        default="data/temporal-sdk-data.json",
        help="Path to a JSON file with documents (default: data/temporal-sdk-data.json)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of documents to send per /ingest call (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip the /state/load step (no simulated restart)",
    )
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    data_path = Path(args.data)

    # ---- 1. Load documents from file ----------------------------------------
    print(f"1. Loading documents from {data_path}...")
    documents = load_documents(data_path)
    print(f"   Found {len(documents)} documents.\n")

    # ---- 2. Ingest in batches ------------------------------------------------
    total_entities = 0
    total_resolved = 0
    batches = [documents[i : i + args.batch_size] for i in range(0, len(documents), args.batch_size)]
    print(f"2. Ingesting {len(documents)} documents in {len(batches)} batch(es) (POST /ingest)...")
    for idx, batch in enumerate(batches, start=1):
        resp = post_json(f"{base}/ingest", {"documents": batch})
        total_entities += resp.get("entities_extracted", 0)
        total_resolved += resp.get("entities_resolved", 0)
        print(f"   Batch {idx}/{len(batches)}: {resp['documents_processed']} docs, "
              f"{resp['entities_extracted']} entities, {resp['graph_nodes']} nodes, "
              f"{resp['graph_edges']} edges")
    print(f"   Totals: {len(documents)} docs ingested, {total_entities} entities extracted, "
          f"{total_resolved} resolved.")
    print("   (Server auto-saves state and exports graph after each batch.)\n")

    # ---- 3. Explicit save & export -------------------------------------------
    print("3. Explicit save and export (POST /state/save, POST /graph/export)...")
    save_resp = post_no_body(f"{base}/state/save")
    print(f"   State saved: {save_resp}")
    export_resp = post_no_body(f"{base}/graph/export")
    print(f"   Graph exported: {export_resp}\n")

    # ---- 4. (Optional) Simulate restart via /state/load ----------------------
    if not args.skip_load:
        print("4. Simulating restart: load state (POST /state/load)...")
        load_resp = post_no_body(f"{base}/state/load")
        print(f"   State loaded: {load_resp}\n")

    # ---- 5. Answer a query ---------------------------------------------------
    step = "5" if not args.skip_load else "4"
    queries = [
        "What are Temporal Workflows?",
        "How does Nexus connect namespaces?",
        "What is a Worker in Temporal?",
    ]
    print(f"{step}. Answering sample queries (POST /answer)...")
    for query in queries:
        resp = post_json(f"{base}/answer", {"query": query, "top_k": 3})
        answer_text = resp.get("answer", "")[:200]
        citations = len(resp.get("citations", []))
        print(f"\n   Q: {query}")
        print(f"   A: {answer_text}...")
        print(f"   Citations: {citations}")

    print("\nDone. Persisted files are under DATA_DIR (e.g. ./data/state.pkl and ./data/graph/).")


if __name__ == "__main__":
    main()
