# tutor/concept_compressor.py

from typing import List, Dict


def compress_concepts(
    retrieved_concepts: List[Dict],
    max_concepts: int = 2
) -> List[Dict]:
    """
    Compress retrieved concepts into a focused teaching set.

    Purpose:
    - Select a primary concept for explanation
    - Optionally include supporting concepts
    - Remove semantic noise

    Args:
        retrieved_concepts (List[Dict]):
            Output from retrieve_relevant_concepts(), each dict contains:
            {
                "subject": str,
                "chapter": str,
                "concept": str,
                "score": float,
                "questions": List[str],
                "common_traps": List[str]
            }

        max_concepts (int):
            Maximum number of concepts to retain (default: 2)

    Returns:
        List[Dict]:
            Compressed list of concepts ordered by teaching priority
    """

    if not retrieved_concepts:
        return []

    # ----------------------------------------------------
    # 1️⃣ Sort by semantic relevance (lower score = closer)
    # ----------------------------------------------------
    sorted_concepts = sorted(
        retrieved_concepts,
        key=lambda x: x.get("score", float("inf"))
    )

    # ----------------------------------------------------
    # 2️⃣ Select primary concept
    # ----------------------------------------------------
    primary = sorted_concepts[0]

    compressed = [primary]

    # ----------------------------------------------------
    # 3️⃣ Select supporting concepts (same subject & chapter)
    # ----------------------------------------------------
    for concept in sorted_concepts[1:]:
        if len(compressed) >= max_concepts:
            break

        # Keep concepts only if they are pedagogically aligned
        if (
            concept["subject"] == primary["subject"]
            and concept["chapter"] == primary["chapter"]
            and concept["concept"] != primary["concept"]
        ):
            compressed.append(concept)

    return compressed