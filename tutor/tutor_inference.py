"""
Tutor inference pipeline.

Given a student question:
  1. retrieve the most relevant concept(s) and real past-year questions
  2. assemble a single prompt (concept context + candidate PYQs)
  3. make one structured LLM call for explanation + traps + practice picks
  4. resolve the model's recommended qids back to full questions (text + solution)

Returns a plain dict that the UI layer can render directly.
"""

import sys
from pathlib import Path

# Allow running this file directly as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.concept_retriever import (
    retrieve_relevant_concepts,
    retrieve_relevant_pyqs,
    get_pyqs_by_ids,
)
from tutor.concept_compressor import compress_concepts
from tutor.tutor_prompt_template import build_tutor_prompt
from llm.llm_client import generate_tutor_response

# Each candidate now carries its full verified solution into the prompt, so keep
# the set small to bound token cost. The model teaches the best 2-3 of these.
MAX_CANDIDATE_PYQS = 4


def _collect_candidate_pyqs(query: str, concept: dict) -> list:
    """
    Gather candidate practice questions for the model to choose from:
    the concept's curated linked PYQs plus questions semantically matched to
    the query. Deduplicated by qid, capped at MAX_CANDIDATE_PYQS.
    """
    candidates = {}

    # Curated questions linked to this concept in the graph.
    for pyq in get_pyqs_by_ids(concept.get("linked_questions", [])):
        candidates[pyq["qid"]] = pyq

    # Semantically matched questions (same subject as the primary concept).
    for pyq in retrieve_relevant_pyqs(query, k=5, subject=concept.get("subject")):
        candidates.setdefault(pyq["qid"], pyq)

    return list(candidates.values())[:MAX_CANDIDATE_PYQS]


def tutor(student_query: str, k: int = 5, subject: str = None) -> dict:
    """
    Produce a full tutor response for `student_query`.

    Args:
        student_query: the student's natural-language question.
        k:             number of concepts to retrieve before compression.
        subject:       optional subject filter ("Physics"/"Chemistry"/"Mathematics").

    Returns:
        dict with the primary concept, explanation, traps, and practice
        problems (each carrying the full resolved question), or an
        {"error": ...} dict if nothing relevant was retrieved.
    """
    retrieved = retrieve_relevant_concepts(student_query, k=k, subject=subject)
    compressed = compress_concepts(retrieved, max_concepts=2)

    if not compressed:
        return {
            "query": student_query,
            "error": "No relevant concept was found for this question. "
                     "Try rephrasing, or pick the subject explicitly.",
        }

    concept = compressed[0]
    candidate_pyqs = _collect_candidate_pyqs(student_query, concept)

    prompt = build_tutor_prompt(student_query, concept, candidate_pyqs)
    result = generate_tutor_response(prompt)

    # Resolve each worked example's qid back to the full question (text, options,
    # answer, verified solution) so the UI can show the ground-truth solution
    # alongside the model's step-by-step walkthrough.
    by_qid = {q["qid"]: q for q in candidate_pyqs}
    worked_examples = []
    for ex in result.worked_examples:
        worked_examples.append({
            "qid": ex.qid,
            "what_it_tests": ex.what_it_tests,
            "approach": ex.approach,
            "steps": ex.steps,
            "trap": ex.trap,
            "question": by_qid.get(ex.qid),
        })

    return {
        "query": student_query,
        "primary_concept": concept,
        "supporting_concepts": compressed[1:],
        "concept_explanation": result.concept_explanation,
        "common_traps": [t.model_dump() for t in result.common_traps],
        "worked_examples": worked_examples,
        "candidate_pyqs": candidate_pyqs,
    }


# Backwards-compatible alias for the previous entry point name.
def chunked_tutor_inference(student_query: str, k: int = 5):
    return tutor(student_query, k=k)


if __name__ == "__main__":
    query = "Why does the equilibrium constant decrease with temperature for exothermic reactions?"
    answer = tutor(query, k=5)
    print("\n📘 AI Tutor Response\n")
    if "error" in answer:
        print(answer["error"])
    else:
        print("CONCEPT:", answer["primary_concept"]["concept"])
        print("\nEXPLANATION:\n", answer["concept_explanation"])
        print("\nTRAPS:")
        for t in answer["common_traps"]:
            print(" -", t["trap"])
        print("\nWORKED EXAMPLES:")
        for ex in answer["worked_examples"]:
            print(f" - {ex['qid']} → {ex['what_it_tests']} ({len(ex['steps'])} steps)")
