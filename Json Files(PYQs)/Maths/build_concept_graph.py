import json
from collections import defaultdict
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
SCRIPT_DIR = Path(__file__).parent
INPUT_FILE = SCRIPT_DIR / "Mathematics_PYQs.json"
OUTPUT_FILE = SCRIPT_DIR / "concept_graph_Maths.json"

# -----------------------------
# HELPER STRUCTURES
# -----------------------------
def new_concept_node(subject, chapter):
    return {
        "subject": subject,
        "chapter": chapter,
        "subconcepts": {}
    }

def new_subconcept_node():
    return {
        "formulas": {},
        "questions": [],
        "common_traps": [],
        "prerequisites": []  # to be enriched later
    }

def new_formula_node(expression):
    return {
        "expression": expression,
        "used_in_questions": []
    }

# -----------------------------
# MAIN GRAPH BUILDER
# -----------------------------
def build_concept_graph(questions):
    graph = defaultdict(lambda: defaultdict(dict))

    for q in questions:
        subject = q["subject"]
        chapter = q["chapter"]
        qid = q["qid"]

        # Initialize chapter node
        if "node" not in graph[subject][chapter]:
            graph[subject][chapter]["node"] = new_concept_node(subject, chapter)

        chapter_node = graph[subject][chapter]["node"]

        concept_tags = q.get("concept_tags", [])
        formula_tags = q.get("formula_tags", [])
        raw_traps = q.get("common_traps", [])

        # Normalize common_traps into a clean list
        if isinstance(raw_traps, str):
            traps = [raw_traps.strip()] if raw_traps.strip() else []
        elif isinstance(raw_traps, list):
            traps = [t.strip() for t in raw_traps if isinstance(t, str) and t.strip()]
        else:
            traps = []

        for concept in concept_tags:
            # Initialize subconcept
            if concept not in chapter_node["subconcepts"]:
                chapter_node["subconcepts"][concept] = new_subconcept_node()

            subconcept_node = chapter_node["subconcepts"][concept]

            # Link question
            if qid not in subconcept_node["questions"]:
                subconcept_node["questions"].append(qid)

            # Add traps
            for trap in traps:
                if trap not in subconcept_node["common_traps"]:
                    subconcept_node["common_traps"].append(trap)

            # Handle formulas
            for formula in formula_tags:
                if formula not in subconcept_node["formulas"]:
                    subconcept_node["formulas"][formula] = new_formula_node(formula)

                if qid not in subconcept_node["formulas"][formula]["used_in_questions"]:
                    subconcept_node["formulas"][formula]["used_in_questions"].append(qid)

    return graph

# -----------------------------
# SERIALIZATION
# -----------------------------
def serialize_graph(graph):
    output = {
        "metadata": {
            "exam": "JEE Advanced",
            "subjects": list(graph.keys())
        },
        "concept_graph": {}
    }

    for subject, chapters in graph.items():
        output["concept_graph"][subject] = {}
        for chapter, data in chapters.items():
            output["concept_graph"][subject][chapter] = data["node"]

    return output

# -----------------------------
# ENTRY POINT
# -----------------------------
def main():
    with open(INPUT_FILE, "r") as f:
        questions = json.load(f)

    graph = build_concept_graph(questions)
    graph_json = serialize_graph(graph)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(graph_json, f, indent=2)

    print(f"✅ Concept graph generated: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()