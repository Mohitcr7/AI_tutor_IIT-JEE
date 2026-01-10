"""
This module converts a concept graph structure into LangChain Document objects.
It extracts concepts from the hierarchical graph (Subject -> Chapter -> Concept)
and creates documents with the concept information and metadata for embedding.
"""

def build_concept_documents(concept_graph):
    """
    Convert a concept graph into a list of document dictionaries.
    
    This function traverses the hierarchical structure of the concept graph:
    - Subject level: Physics, Chemistry, Maths
    - Chapter level: Topics within each subject
    - Concept level: Individual concepts with formulas, traps, and questions
    
    Each concept is converted into a document containing:
    - page_content: Formatted text with concept details
    - metadata: Subject, chapter, concept name, and related questions
    
    Args:
        concept_graph (dict): A nested dictionary with structure:
            {
                "concept_graph": {
                    "subject": {
                        "chapter": {
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
    
    Returns:
        list: List of document dictionaries with 'page_content' and 'metadata' keys
    """
    # List to store all processed documents
    docs = []

    # Iterate through each subject (Physics, Chemistry, Maths)
    for subject, chapters in concept_graph["concept_graph"].items():
        # Iterate through each chapter within the subject
        for chapter, chapter_node in chapters.items():
            # Iterate through each concept within the chapter
            for concept, data in chapter_node["subconcepts"].items():
                # Format the concept information as readable text
                # This text will be embedded and used for semantic search
                text = f"""
                Concept: {concept.replace('_', ' ')}
                Subject: {subject}
                Chapter: {chapter}
                Key formulas: {', '.join(data.get('formulas', {}).keys())}
                Common traps: {', '.join(data.get('common_traps', []))}
                """

                # Create a document dictionary with content and metadata
                docs.append({
                    "page_content": text.strip(),  # The text that will be embedded
                    "metadata": {
                        "subject": subject,
                        "chapter": chapter,
                        "concept": concept,
                        "questions": ",".join(data.get("questions", [])),
                        "common_traps": " | ".join(data.get("common_traps", []))
                    }
                })

    return docs