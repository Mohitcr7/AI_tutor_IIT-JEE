def build_concept_documents(concept_graph):
    documents = []

    for subject, chapters in concept_graph["concept_graph"].items():
        for chapter, chapter_node in chapters.items():
            for concept, data in chapter_node["subconcepts"].items():
                text = f"""
                Concept: {concept.replace('_', ' ')}
                Subject: {subject}
                Chapter: {chapter}
                Formulas: {', '.join(data.get('formulas', {}).keys())}
                Common traps: {', '.join(data.get('common_traps', []))}
                """

                documents.append({
                    "id": f"{subject}|{chapter}|{concept}",
                    "subject": subject,
                    "chapter": chapter,
                    "concept": concept,
                    "text": text.strip()
                })

    return documents