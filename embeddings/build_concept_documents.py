"""
Convert a concept graph into embeddable documents.

Each concept (Subject -> Chapter -> Concept) becomes one document:
  - `document`: the text that gets embedded for semantic search
  - `metadata`: structured fields used by the tutor after retrieval

ChromaDB metadata values must be primitives (str/int/float/bool), so list
and dict fields are stored as JSON strings and decoded on read by the
retriever. This is what previously broke: traps were stored as a plain
string and then iterated character-by-character downstream.
"""

import json


def build_concept_documents(concept_graph):
    """
    Flatten a concept graph into a list of {id, document, metadata} dicts.

    Args:
        concept_graph (dict): parsed concept graph JSON with a top-level
            "concept_graph" key mapping Subject -> Chapter -> {subconcepts: ...}.

    Returns:
        list[dict]: one entry per concept, each with:
            - id (str):        stable unique id "Subject|Chapter|concept"
            - document (str):  text to embed
            - metadata (dict): subject, chapter, concept, and JSON-encoded
                               formulas / common_traps / questions
    """
    docs = []

    for subject, chapters in concept_graph["concept_graph"].items():
        for chapter, chapter_node in chapters.items():
            for concept, data in chapter_node.get("subconcepts", {}).items():
                formulas = list(data.get("formulas", {}).keys())
                traps = data.get("common_traps", [])
                questions = data.get("questions", [])

                document = "\n".join([
                    f"Concept: {concept.replace('_', ' ')}",
                    f"Subject: {subject}",
                    f"Chapter: {chapter}",
                    f"Key formulas: {', '.join(formulas)}",
                    f"Common traps: {', '.join(traps)}",
                ])

                docs.append({
                    "id": f"{subject}|{chapter}|{concept}",
                    "document": document,
                    "metadata": {
                        "subject": subject,
                        "chapter": chapter,
                        "concept": concept,
                        # JSON-encoded so lists survive Chroma's primitive-only
                        # metadata; the retriever decodes these back to lists.
                        "formulas": json.dumps(formulas, ensure_ascii=False),
                        "common_traps": json.dumps(traps, ensure_ascii=False),
                        "questions": json.dumps(questions, ensure_ascii=False),
                    },
                })

    return docs
