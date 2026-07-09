"""
Prompt construction for the tutor.

A single prompt drives one structured LLM call that returns the concept
explanation, common traps, and recommended practice problems together. The
practice problems are *selected from real retrieved PYQs* (by qid), not
invented, so the app can show the actual question text and worked solution.

The system prompt lives in llm.llm_client (single source of truth); this module
only assembles the per-query user prompt.
"""


def _format_concept(concept: dict) -> str:
    formulas = concept.get("formulas") or []
    traps = concept.get("common_traps") or []
    lines = [
        f"Concept: {concept.get('concept', '').replace('_', ' ')}",
        f"Subject: {concept.get('subject', '')}",
        f"Chapter: {concept.get('chapter', '')}",
    ]
    if formulas:
        lines.append("Key formulas: " + "; ".join(formulas))
    if traps:
        lines.append("Known traps for this concept:")
        lines.extend(f"  - {t}" for t in traps)
    return "\n".join(lines)


def _format_candidate_pyqs(pyqs: list) -> str:
    if not pyqs:
        return "(none available — recommend practice by describing problem types instead)"
    blocks = []
    for q in pyqs:
        parts = [
            f"[{q.get('qid')}] ({q.get('difficulty_level', 'Unknown')} difficulty, "
            f"{q.get('chapter', '')})",
            f"QUESTION: {q.get('question_text', '').strip()}",
        ]
        if q.get("correct_answer"):
            parts.append(f"CORRECT ANSWER: {q['correct_answer']}")
        if q.get("solution_text"):
            parts.append(f"VERIFIED SOLUTION:\n{q['solution_text'].strip()}")
        blocks.append("\n".join(parts))
    return "\n\n---\n\n".join(blocks)


def build_tutor_prompt(student_query: str, concept: dict, candidate_pyqs: list) -> str:
    """
    Build the single user prompt for a full tutor response.

    Args:
        student_query:  the student's natural-language question.
        concept:        the primary retrieved concept (from the concept graph).
        candidate_pyqs: real retrieved PYQs the model may recommend, each with
                        at least `qid`, `question_text`, `chapter`, `difficulty_level`.
    """
    return f"""A student preparing for JEE Advanced asked:

"{student_query}"

The most relevant concept from the curriculum is:

{_format_concept(concept)}

Candidate past-year questions, each with its official VERIFIED SOLUTION. You may
teach only from these, by their exact IDs:

{_format_candidate_pyqs(candidate_pyqs)}

Produce a complete tutor response with three parts:

1. concept_explanation — Explain the underlying concept clearly and intuitively,
   grounded in the student's question. Focus on the "why". Use exam-relevant
   language. Do not solve the practice problems here.

2. common_traps — The mistakes students actually make on this concept in the
   exam. For each, give the trap, why students fall for it, and the correction.
   Prefer the known traps listed above, but add others if they are important.

3. worked_examples — Choose the 2 candidate questions (at most 3) that best
   reinforce this concept, and teach each one. For each, provide:
     - what_it_tests: one line on why it reinforces the concept
     - approach: the first insight or how to start — a hint that gets the student
       moving without giving away the whole solution
     - steps: a step-by-step walkthrough that FOLLOWS the verified solution above,
       explaining the reasoning at each step. Teach through the provided solution;
       do not re-derive it or change the method. If the verified solution looks
       wrong, say so in the steps instead of inventing a different one.
     - trap: the specific mistake to avoid on this particular problem.
   Reference each question by its exact qid. If no candidate is suitable, return
   an empty list.
"""
