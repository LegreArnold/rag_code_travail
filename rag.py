import os
import json
import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Charger les variables d'environnement (.env)
load_dotenv()

# Charger l'index FAISS et les métadonnées depuis le disque
def load_index(folder):
    index = faiss.read_index(f"{folder}/faiss.index")
    with open(f"{folder}/chunks_meta.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Index loaded — {index.ntotal} vectors, {len(chunks)} chunks")
    return index, chunks

# Rechercher les k chunks les plus pertinents pour une question
def search(question, model, index, chunks, k=4):
    # Étape 1 : transformer la question en vecteur
    question_vector = model.encode([question]).astype(np.float32)
    
    # Étape 2 : normaliser (obligatoire car on a utilisé IndexFlatIP: similarité cosinus)
    faiss.normalize_L2(question_vector)
    
    # Étape 3 : rechercher les k vecteurs les plus proches
    scores, indices = index.search(question_vector, k)
    
    # Étape 4 : récupérer les chunks correspondants
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "content": chunks[idx]["content"],
            "metadata": chunks[idx]["metadata"],
            "score": float(score)
        })
    
    return results

# Construire le contexte à partir des chunks pertinents
def build_context(results):
    context = "=== Articles du Code du travail pertinents ===\n\n"
    for i, result in enumerate(results, 1):
        meta = result["metadata"]
        context += (
            f"[Source {i}] Article {meta['article']} — {meta['titre']}\n"
            f"{result['content']}\n\n"
        )
    return context

# Générer la réponse via l'API Groq
def generate_response(client, question, results):
    context = build_context(results)
    
    system_prompt = """Tu es un assistant juridique spécialisé dans le Code du travail français.
Tu réponds UNIQUEMENT en te basant sur les articles fournis dans le contexte.
Règles :
1. Cite toujours le numéro d'article (ex: "Selon l'article L3121-27...")
2. Si la réponse n'est pas dans le contexte, dis-le clairement
3. Termine TOUJOURS par : "Cet assistant ne fournit pas de conseil juridique. Consultez un avocat ou l'inspection du travail pour votre situation personnelle."
"""

    user_prompt = f"{context}\nQuestion : {question}"
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=1000
    )
    
    return response.choices[0].message.content


# Point d'entrée principal — boucle interactive de questions-réponses
def main():
    print("=" * 50)
    print(" ASSISTANT CODE DU TRAVAIL")
    print("=" * 50)

    # Vérification de la clé API
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Erreur : clé API Groq manquante dans le fichier .env")
        return

    # Chargement de l'index et du modèle
    print("\nLoading knowledge base...")
    index, chunks = load_index("index")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    client = Groq(api_key=api_key)

    print("\n System ready. Type 'quit' to exit.\n")
    print("-" * 50)

    # Boucle interactive
    while True:
        question = input("\n Your question : ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Au revoir !")
            break

        if not question:
            continue

        # Étape 1 : recherche des chunks pertinents
        print("\n Searching...")
        results = search(question, model, index, chunks, k=4)

        # Étape 2 : génération de la réponse
        print(" Generating response...\n")
        response = generate_response(client, question, results)

        # Étape 3 : affichage
        print("-" * 50)
        print(response)
        print("\n Sources :")
        for i, result in enumerate(results, 1):
            meta = result["metadata"]
            print(f"  [{i}] Art. {meta['article']} — {meta['titre']} (score : {result['score']:.2f})")
        print("-" * 50)


if __name__ == "__main__":
    main()