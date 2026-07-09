"""
Semantic retrieval over the persisted ChromaDB store.

Exposes retrieval for both collections:
  - retrieve_relevant_concepts(query): concept graph nodes
  - retrieve_relevant_pyqs(query):     real past-year questions with solutions
  - get_pyqs_by_ids(qids):             look up specific questions by qid

The Chroma client and embedding model are created once at import time and
reused for every query (previously the embedding model was reloaded on each
call, adding seconds of latency per request).
"""

import json
from pathlib import Path
from typing import Dict, List

import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = Path(__file__).resolve().parents[1]
CHROMA_DIR = BASE_DIR / "vectorstore" / "chroma_concepts"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CONCEPTS_COLLECTION = "jee_concepts"
PYQS_COLLECTION = "jee_pyqs"

# Created once, reused for all queries.
_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME
)
_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_concepts = _client.get_collection(CONCEPTS_COLLECTION, embedding_function=_embedding_fn)
_pyqs = _client.get_collection(PYQS_COLLECTION, embedding_function=_embedding_fn)


def _loads(value):
    """Decode a JSON-encoded metadata field back to a list; tolerate plain strings."""
    if not value:
        return []
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return [value]


def _decode_concept(metadata: dict, distance: float) -> Dict:
    return {
        "subject": metadata.get("subject"),
        "chapter": metadata.get("chapter"),
        "concept": metadata.get("concept"),
        "formulas": _loads(metadata.get("formulas")),
        "common_traps": _loads(metadata.get("common_traps")),
        "linked_questions": _loads(metadata.get("questions")),
        # Cosine distance: lower is closer. `similarity` is the friendlier 0..1 form.
        "distance": round(float(distance), 4),
        "similarity": round(1.0 - float(distance), 4),
    }


def _decode_pyq(metadata: dict, distance: float = None) -> Dict:
    pyq = {
        "qid": metadata.get("qid"),
        "subject": metadata.get("subject"),
        "chapter": metadata.get("chapter"),
        "year": metadata.get("year"),
        "difficulty_level": metadata.get("difficulty_level"),
        "question_text": metadata.get("question_text"),
        "correct_answer": metadata.get("correct_answer"),
        "solution_text": metadata.get("solution_text"),
        "options": _loads(metadata.get("options")),
        "concept_tags": _loads(metadata.get("concept_tags")),
        "formula_tags": _loads(metadata.get("formula_tags")),
        "common_traps": _loads(metadata.get("common_traps")),
    }
    if distance is not None:
        pyq["distance"] = round(float(distance), 4)
        pyq["similarity"] = round(1.0 - float(distance), 4)
    return pyq


def retrieve_relevant_concepts(query: str, k: int = 5, subject: str = None) -> List[Dict]:
    """Return the top-k concept graph nodes most relevant to `query`."""
    where = {"subject": subject} if subject else None
    results = _concepts.query(query_texts=[query], n_results=k, where=where)
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return [_decode_concept(m, d) for m, d in zip(metadatas, distances)]


def retrieve_relevant_pyqs(query: str, k: int = 4, subject: str = None) -> List[Dict]:
    """Return the top-k past-year questions most relevant to `query`."""
    where = {"subject": subject} if subject else None
    results = _pyqs.query(query_texts=[query], n_results=k, where=where)
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return [_decode_pyq(m, d) for m, d in zip(metadatas, distances)]


def get_pyqs_by_ids(qids: List[str]) -> List[Dict]:
    """Fetch specific questions by qid (order follows Chroma's response)."""
    qids = [q for q in qids if q]
    if not qids:
        return []
    results = _pyqs.get(ids=qids)
    return [_decode_pyq(m) for m in results["metadatas"]]


if __name__ == "__main__":
    q = "Why does entropy increase in free expansion of gas?"
    print(f"Query: {q}\n")
    print("── Concepts ──")
    for r in retrieve_relevant_concepts(q, k=3):
        print(f"  {r['concept']} ({r['subject']} / {r['chapter']})  sim={r['similarity']}")
    print("\n── PYQs ──")
    for r in retrieve_relevant_pyqs(q, k=3):
        print(f"  {r['qid']} ({r['chapter']})  sim={r['similarity']}")
        print(f"    {r['question_text'][:90]}...")
