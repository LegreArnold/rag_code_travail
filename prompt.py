# prompt.py
# Contient le prompt système et la fonction de construction du contexte

SYSTEM_PROMPT = """Tu es un assistant juridique spécialisé dans le Code du travail français.
Tu réponds UNIQUEMENT en te basant sur les articles fournis dans le contexte.
Règles :
1. Cite toujours le numéro d'article (ex: "Selon l'article L3121-27...")
2. Si la réponse n'est pas dans le contexte, dis-le clairement
3. Termine TOUJOURS par : "Cet assistant ne fournit pas de conseil juridique. Consultez un avocat ou l'inspection du travail pour votre situation personnelle."
"""

def build_context(results):
    context = "=== Articles du Code du travail pertinents ===\n\n"
    for i, result in enumerate(results, 1):
        meta = result["metadata"]
        context += (
            f"[Source {i}] Article {meta['article']} — {meta['titre']}\n"
            f"{result['content']}\n\n"
        )
    return context