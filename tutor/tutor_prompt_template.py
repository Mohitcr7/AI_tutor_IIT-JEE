def build_concept_prompt(student_query, concept):
    return f"""
You are an IIT-JEE tutor.

Explain the following concept clearly and intuitively.

Student Query:
{student_query}

Concept:
{concept['concept']}

Chapter:
{concept['chapter']} ({concept['subject']})

Instructions:
- Explain the idea in simple but precise terms
- Use exam-relevant language
- Avoid solving problems
- Avoid mentioning traps or mistakes
- Keep explanation concise but complete
"""


def build_trap_prompt(concept):
    traps = concept.get("common_traps", [])

    if not traps:
        traps_text = "No common traps found for this concept."
    else:
        traps_text = "\n".join(f"- {t}" for t in traps)
    return f"""
You are an IIT-JEE expert.

Based on the concept below, explain the most common mistakes students make.

Concept:
{concept['concept']}

Common Traps:
{traps_text}

Instructions:
- Explain WHY students make these mistakes
- Link mistakes to incorrect intuition
- Mention how JEE frames questions to exploit these traps
- Do NOT explain the concept again
"""


def build_practice_prompt(concept):
    questions = concept.get("questions", [])

    if not questions:
        questions_text = "No linked JEE questions found. Generate similar practice problems instead."
    else:
        questions_text = ", ".join(questions)
    return f"""
You are an IIT-JEE problem setter.

Using the concept below, suggest practice problems.

Concept:
{concept['concept']}

Reference PYQs:
{questions_text}

Instructions:
- Suggest 2–3 practice problems
- Vary difficulty (easy → JEE Advanced)
- Mention what concept each problem tests
- Do NOT solve the problems
"""