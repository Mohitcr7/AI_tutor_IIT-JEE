import sys
from pathlib import Path

# Add project root to Python path to enable absolute imports
# This allows the script to work regardless of where it's executed from
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.concept_retriever import retrieve_relevant_concepts
from tutor.concept_compressor import compress_concepts
from tutor.tutor_prompt_template import (
    build_concept_prompt,
    build_trap_prompt,
    build_practice_prompt
)
from llm.llm_client import generate_response


def chunked_tutor_inference(student_query: str, k: int = 5):

    # 1️⃣ Retrieve concepts
    retrieved = retrieve_relevant_concepts(student_query, k=k)
    compressed = compress_concepts(retrieved, max_concepts=2)

    concept = compressed[0]  # Focus on strongest concept

    # 2️⃣ Concept Explanation
    concept_prompt = build_concept_prompt(student_query, concept)
    concept_response = generate_response(concept_prompt)

    # 3️⃣ Trap Explanation
    trap_prompt = build_trap_prompt(concept)
    trap_response = generate_response(trap_prompt)

    # 4️⃣ Practice Problems
    practice_prompt = build_practice_prompt(concept)
    practice_response = generate_response(practice_prompt)

    # 5️⃣ Assemble Tutor Output
    return {
        "concept_explanation": concept_response,
        "common_traps": trap_response,
        "practice_problems": practice_response
    }
    

if __name__ == "__main__":
    query = "Why does equilibrium constant decrease with temperature for exothermic reactions?"

    answer = chunked_tutor_inference(query, k=5)
    print("\n📘 AI Tutor Response:\n")
    print(answer)