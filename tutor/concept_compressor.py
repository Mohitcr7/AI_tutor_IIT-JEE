# tutor/concept_compressor.py

from typing import Dict, List


def compress_concepts(
    retrieved_concepts: List[Dict],
    max_concepts: int = 2,
) -> List[Dict]:
    """
    Reduce retrieved concepts to a focused teaching set.

    Picks the closest-matching concept as the primary, then adds up to
    `max_concepts - 1` supporting concepts from the same subject and chapter.

    Args:
        retrieved_concepts: output of retrieve_relevant_concepts(); each dict
            carries a `distance` field (cosine distance, lower is closer).
        max_concepts: maximum number of concepts to keep.

    Returns:
        Concepts ordered by teaching priority (primary first).
    """
    if not retrieved_concepts:
        return []

    # Rank by semantic distance (lower = closer). The retriever populates
    # `distance`; fall back to +inf so any mis-shaped entry sorts last.
    ranked = sorted(retrieved_concepts, key=lambda c: c.get("distance", float("inf")))

    primary = ranked[0]
    compressed = [primary]

    for concept in ranked[1:]:
        if len(compressed) >= max_concepts:
            break
        if (
            concept["subject"] == primary["subject"]
            and concept["chapter"] == primary["chapter"]
            and concept["concept"] != primary["concept"]
        ):
            compressed.append(concept)

    return compressed
