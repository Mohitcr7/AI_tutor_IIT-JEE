"""
Convert the JEE PYQ corpus into embeddable documents.

The aggregated corpus files (one per subject) hold the full question text,
options, correct answer, worked solution, and per-question tags. Each question
becomes one document whose embedded text is the question statement plus its
concept tags and chapter, so semantic search matches on what the question is
actually about. The solution and answer live in metadata and are surfaced to
the student after retrieval.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PYQ_DIR = BASE_DIR / "Json Files(PYQs)"

# Aggregated corpus file per subject (full text + solutions).
PYQ_FILES = {
    "Physics": PYQ_DIR / "Physics" / "Physics_PYQs.json",
    "Chemistry": PYQ_DIR / "Chemistry" / "chemistry_PYQs.json",
    "Mathematics": PYQ_DIR / "Maths" / "Mathematics_PYQs.json",
}


def _as_json(value):
    """Serialize list/dict metadata to a JSON string (Chroma stores primitives only)."""
    return json.dumps(value or [], ensure_ascii=False)


def build_pyq_documents():
    """
    Load every aggregated PYQ file and flatten to {id, document, metadata} dicts.

    Deduplicates by `qid` (the corpus files contain some repeated questions),
    keeping the first occurrence.

    Returns:
        list[dict]: one entry per unique question, each with:
            - id (str):        the qid, e.g. "JEE_Adv_2023_Chem_1"
            - document (str):  text to embed (question + tags + chapter)
            - metadata (dict): qid, subject, chapter, year, difficulty,
                               options/answer/solution, and JSON-encoded tags
    """
    docs = []
    seen = set()

    for subject, path in PYQ_FILES.items():
        questions = json.loads(path.read_text(encoding="utf-8"))

        for q in questions:
            qid = q.get("qid")
            if not qid or qid in seen:
                continue
            seen.add(qid)

            concept_tags = q.get("concept_tags", [])
            chapter = q.get("chapter", "")

            document = "\n".join([
                f"Question: {q.get('question_text', '')}",
                f"Chapter: {chapter}",
                f"Concepts: {', '.join(concept_tags)}",
            ])

            docs.append({
                "id": qid,
                "document": document,
                "metadata": {
                    "qid": qid,
                    "subject": q.get("subject", subject),
                    "chapter": chapter,
                    "year": str(q.get("year", "")),
                    "difficulty_level": q.get("difficulty_level", ""),
                    "question_text": q.get("question_text", ""),
                    "correct_answer": q.get("correct_answer", ""),
                    "solution_text": q.get("solution_text", ""),
                    "options": _as_json(q.get("options", [])),
                    "concept_tags": _as_json(concept_tags),
                    "formula_tags": _as_json(q.get("formula_tags", [])),
                    "common_traps": _as_json(q.get("common_traps", [])),
                },
            })

    return docs


if __name__ == "__main__":
    built = build_pyq_documents()
    print(f"Built {len(built)} unique PYQ documents")
    by_subject = {}
    for d in built:
        by_subject[d["metadata"]["subject"]] = by_subject.get(d["metadata"]["subject"], 0) + 1
    for subj, n in sorted(by_subject.items()):
        print(f"  {subj}: {n}")
