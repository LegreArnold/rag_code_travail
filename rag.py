import os
from prompt import SYSTEM_PROMPT, build_context
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
    # Transformer la question en vecteur
    question_vector = model.encode([question]).astype(np.float32)
    
    # Normaliser (obligatoire car on a utilisé IndexFlatIP : similarité cosinus)
    faiss.normalize_L2(question_vector)
    
    # Rechercher les k vecteurs les plus proches
    scores, indices = index.search(question_vector, k)
    
    # Récupérer les chunks correspondants
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

# Générer la réponse via l'API Groq
def generate_response(client, question, results):
    context = build_context(results)

    user_prompt = f"{context}\nQuestion : {question}"
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=1000
    )
    return response.choices[0].message.content

# Point d'entrée principal — boucle interactive de questions-réponses
def main():
    print("=" * 50)
    print("  ASSISTANT CODE DU TRAVAIL")
    print("=" * 50)

    # Vérification de la clé API
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Erreur : cle API Groq manquante dans le fichier .env")
        return

    # Chargement de l'index et du modele
    print("\nLoading knowledge base...")
    index, chunks = load_index("index")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    client = Groq(api_key=api_key)

    print("\nSystem ready. Type 'quit' to exit.\n")
    print("-" * 50)

    # Boucle interactive
    while True:
        question = input("\nYour question : ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Au revoir !")
            break

        if not question:
            continue

        # Etape 1 : recherche des chunks pertinents
        print("\nSearching...")
        results = search(question, model, index, chunks, k=4)

        # Verification du score de confiance
        best_score = results[0]["score"] if results else 0
        if best_score < 0.45:
            print(f"Warning : no directly relevant article found (best score : {best_score:.2f})")
            print("The response may be imprecise or off-topic.\n")

        # Etape 2 : generation de la reponse
        print("Generating response...\n")
        response = generate_response(client, question, results)

        # Etape 3 : affichage
        print("-" * 50)
        print(response)
        print("\nSources :")
        for i, result in enumerate(results, 1):
            meta = result["metadata"]
            print(f"  [{i}] Art. {meta['article']} — {meta['titre']} (score : {result['score']:.2f})")
        print("-" * 50)


if __name__ == "__main__":
    main()