"""
Build and persist the ChromaDB vector store used by the tutor.

Creates two collections in one persistent store:
  - "jee_concepts": one document per concept graph node (the teaching backbone)
  - "jee_pyqs":     one document per past-year question (real problems + solutions)

Run this once during setup, or whenever the concept graphs or PYQ corpus change:

    python embeddings/build_concept_store.py

Uses chromadb + sentence-transformers directly (no LangChain wrappers), so the
embedding model is configured in exactly one place and the store carries a
cosine-distance HNSW index.
"""

import json
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# Make sibling imports work whether this runs as a script (python
# embeddings/build_concept_store.py) or is imported as embeddings.build_concept_store.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from build_concept_documents import build_concept_documents
from build_pyq_documents import build_pyq_documents

BASE_DIR = Path(__file__).resolve().parents[1]
CHROMA_DIR = BASE_DIR / "vectorstore" / "chroma_concepts"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CONCEPTS_COLLECTION = "jee_concepts"
PYQS_COLLECTION = "jee_pyqs"

# Concept graph files. NOTE: filenames are matched exactly — this is
# case-sensitive on Linux (the deployment target), so the chemistry file is
# lowercase here to match the file on disk.
CONCEPT_GRAPH_FILES = [
    "concept_graph_Physics.json",
    "concept_graph_chemistry.json",
    "concept_graph_Maths.json",
]


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_concept_documents():
    docs = []
    for filename in CONCEPT_GRAPH_FILES:
        graph = _load_json(BASE_DIR / "data" / filename)
        docs.extend(build_concept_documents(graph))
    return docs


def _add_documents(collection, docs, batch_size=256):
    """Embed and insert documents in batches (Chroma computes embeddings here)."""
    for start in range(0, len(docs), batch_size):
        batch = docs[start:start + batch_size]
        collection.add(
            ids=[d["id"] for d in batch],
            documents=[d["document"] for d in batch],
            metadatas=[d["metadata"] for d in batch],
        )


def _reset_collection(client, name, embedding_fn):
    """Drop and recreate a collection via the API for a clean rebuild.

    We deliberately do NOT rmtree the store directory: ChromaDB caches the
    client per path, so deleting the directory under a live client corrupts it
    ("readonly database" / "could not connect to tenant"). Resetting through the
    API keeps the client valid.
    """
    try:
        client.delete_collection(name)
    except Exception:
        pass  # collection may not exist yet
    return client.create_collection(
        name=name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def build_store():
    """(Re)build both collections and persist them to disk."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    concept_docs = _load_concept_documents()
    concepts = _reset_collection(client, CONCEPTS_COLLECTION, embedding_fn)
    _add_documents(concepts, concept_docs)

    pyq_docs = build_pyq_documents()
    pyqs = _reset_collection(client, PYQS_COLLECTION, embedding_fn)
    _add_documents(pyqs, pyq_docs)

    print("✅ Vector store built successfully")
    print(f"📦 Persisted to: {CHROMA_DIR}")
    print(f"📊 Concepts embedded: {len(concept_docs)}")
    print(f"📊 PYQs embedded:     {len(pyq_docs)}")


def ensure_store():
    """
    Build the store only if it is missing (used on first boot in deployment).

    Returns True if a build ran, False if an existing store was reused. This
    lets a fresh clone come up with just the source JSON committed — no need to
    commit the vector store — at the cost of a one-time build on first boot.
    """
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        names = {c.name for c in client.list_collections()}
        if {CONCEPTS_COLLECTION, PYQS_COLLECTION}.issubset(names):
            return False
    except Exception:
        pass
    build_store()
    return True


if __name__ == "__main__":
    build_store()
