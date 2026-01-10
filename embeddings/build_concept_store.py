import json
from pathlib import Path
# LangChain imports for embeddings and vector store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
# Local module for converting concept graphs to documents
from build_concept_documents import build_concept_documents

# ============================================================================
# CONFIGURATION & DIRECTORY SETUP
# ============================================================================

# Get the base directory (parent of 'embeddings' folder)
# This allows the script to work regardless of where it's executed from
BASE_DIR = Path(__file__).resolve().parents[1]

# Define the persistent storage directory for ChromaDB
# Vector store will be saved here for future retrieval without reprocessing
CHROMA_DIR = BASE_DIR / "vectorstore" / "chroma_concepts"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_graph(path):
    """
    Load a concept graph from a JSON file.
    
    The JSON file contains a hierarchical structure of concepts organized by
    subject, chapter, and individual concepts with their properties.
    
    Args:
        path (Path): Absolute path to the JSON file
        
    Returns:
        dict: Parsed concept graph with structure:
              {
                  "concept_graph": {
                      "Subject": {
                          "Chapter": {
                              "subconcepts": {
                                  "concept_name": {
                                      "formulas": {...},
                                      "common_traps": [...],
                                      "questions": [...]
                                  }
                              }
                          }
                      }
                  }
              }
              
    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the JSON file is malformed
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_chroma_store():
    """
    Build and persist a ChromaDB vector store from concept graphs.
    
    This is the main function that orchestrates the entire pipeline:
    
    Process:
    1. Initialize HuggingFace embedding model (all-MiniLM-L6-v2)
       - This model generates 384-dimensional vectors for each concept
       - Lightweight, fast, and free to use (no API costs)
       - Optimal for semantic similarity tasks in education domain
    
    2. Load concept graphs from all three subjects:
       - Physics: concept_graph_Physics.json
       - Chemistry: concept_graph_Chemistry.json
       - Mathematics: concept_graph_Maths.json
    
    3. Convert each concept graph to Document objects:
       - Extract text content (concept name, formulas, traps)
       - Add metadata (subject, chapter, concept name, questions)
       - Documents are structured for semantic search
    
    4. Create ChromaDB vector store:
       - Uses from_documents() to create embeddings locally
       - All documents are embedded using HuggingFace model
       - Each embedding vector is indexed in ChromaDB
    
    5. Persist to disk:
       - Saves all embeddings and metadata to /vectorstore/chroma_concepts
       - Vector store can be reloaded later without re-embedding
       - Reduces latency for future runs
    
    Side Effects:
        - Creates CHROMA_DIR if it doesn't exist
        - Downloads HuggingFace model on first run (one-time cost)
        - Persists data to disk
        
    Raises:
        FileNotFoundError: If concept graph JSON files are missing
        ValueError: If concept graphs have unexpected structure
    """
    
    # ========================================================================
    # STEP 1: Initialize Embedding Model
    # ========================================================================
    # Create a HuggingFace embedding model instance
    # all-MiniLM-L6-v2 is a lightweight model (33M parameters)
    # Produces 384-dimensional vectors for semantic similarity
    # Model will be automatically downloaded on first run and cached
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ========================================================================
    # STEP 2: Aggregate Documents from All Subjects
    # ========================================================================
    
    # Initialize a list to store all Document objects from all subjects
    # This allows us to create a single unified vector store across subjects
    documents = []

    # Subject files to process (in order: Physics, Chemistry, Maths)
    subject_files = [
        "concept_graph_Physics.json",
        "concept_graph_Chemistry.json",
        "concept_graph_Maths.json",
    ]

    # Process each subject's concept graph
    for subject_file in subject_files:
        # Construct the full path to the concept graph JSON file
        graph_path = BASE_DIR / "data" / subject_file
        
        # Load the concept graph from the JSON file
        # Returns nested dict: Subject → Chapter → Subconcepts
        graph = load_graph(graph_path)
        
        # Convert the concept graph into Document objects
        # Returns list of dicts with 'page_content' and 'metadata' keys
        docs = build_concept_documents(graph)

        # ====================================================================
        # STEP 3: Create Document Objects with Metadata
        # ====================================================================
        
        # Iterate through each document dict returned from build_concept_documents
        for doc in docs:
            # Create a LangChain Document object
            # page_content: The text that will be embedded and searchable
            # metadata: Additional information for filtering and context
            document = Document(
                page_content=doc["page_content"],  # Formatted concept text
                metadata=doc["metadata"]  # Subject, chapter, concept name, etc.
            )
            
            # Add the Document object to our aggregated list
            documents.append(document)

    # ========================================================================
    # STEP 4: Generate Embeddings and Create Vector Store
    # ========================================================================
    
    # Create a Chroma vector store from all documents
    # The from_documents() method:
    # - Generates embeddings for each document using HuggingFace model locally
    # - Stores embeddings and metadata in ChromaDB
    # - Returns a ChromaDB instance for querying
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(CHROMA_DIR),  # Directory to save vector store
        collection_name="jee_concepts"  # Logical name for the document collection
    )

    # ========================================================================
    # STEP 5: Persist Vector Store to Disk
    # ========================================================================
    
    # Explicitly persist the vector store to disk
    # This saves all embeddings and metadata for future use
    # Future runs can load from disk instead of re-embedding
    vectorstore.persist()
    
    # Confirm successful completion
    print("✅ ChromaDB concept store built successfully")
    print(f"📦 Vector store persisted to: {CHROMA_DIR}")
    print(f"📊 Total documents embedded: {len(documents)}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Script entry point when executed directly.
    
    Calls build_chroma_store() to build and persist the vector database.
    This should be run once during setup or whenever concept graphs are updated.
    """
    build_chroma_store()