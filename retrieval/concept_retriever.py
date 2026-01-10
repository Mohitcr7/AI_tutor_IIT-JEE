from pathlib import Path
from typing import List, Dict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]
CHROMA_DIR = BASE_DIR / "vectorstore" / "chroma_concepts"

COLLECTION_NAME = "jee_concepts"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# =============================================================================
# INITIALIZE VECTOR STORE (LOAD FROM DISK)
# =============================================================================

def load_vectorstore():
    """
    Load the persisted ChromaDB vector store.
    This does NOT re-embed data.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )

    return vectorstore


# =============================================================================
# CORE RETRIEVAL FUNCTION
# =============================================================================

def retrieve_relevant_concepts(query: str, k: int = 5) -> List[Dict]:
    """
    Retrieve top-k semantically relevant concepts for a user query.

    Args:
        query (str): Natural language student query
        k (int): Number of relevant concepts to retrieve

    Returns:
        List[Dict]: List of concept dictionaries with:
            - subject
            - chapter
            - concept
            - formulas
            - common_traps
            - linked_questions
            - similarity_score
            - raw_text (for debugging / inspection)
    """

    vectorstore = load_vectorstore()

    # Perform similarity search with scores
    results = vectorstore.similarity_search_with_score(
        query=query,
        k=k
    )

    retrieved_concepts = []

    for doc, score in results:
        metadata = doc.metadata

        retrieved_concepts.append({
            "subject": metadata.get("subject"),
            "chapter": metadata.get("chapter"),
            "concept": metadata.get("concept"),
            "formulas": metadata.get("formulas", []),
            "common_traps": metadata.get("common_traps", []),
            "linked_questions": metadata.get("questions", []),
            "similarity_score": round(float(score), 4),
            "raw_text": doc.page_content
        })

    return retrieved_concepts


# =============================================================================
# QUICK TEST (OPTIONAL)
# =============================================================================

if __name__ == "__main__":
    query = "Why does entropy increase in free expansion of gas?"
    results = retrieve_relevant_concepts(query, k=3)

    for i, r in enumerate(results, 1):
        print(f"\n🔹 Result {i}")
        print(f"Subject   : {r['subject']}")
        print(f"Chapter   : {r['chapter']}")
        print(f"Concept   : {r['concept']}")
        print(f"Score     : {r['similarity_score']}")
        print(f"Traps     : {r['common_traps']}")
        print(f"Questions : {r['linked_questions']}")