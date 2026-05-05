import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


# Charger le corpus JSON
def load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        documents = json.load(f)
    print(f"Corpus loaded : {len(documents)} articles")
    return documents

# Découper le texte en morceaux avec chevauchement
def chunk_text(text, max_size=500, overlap=50):
    if len(text) <= max_size:
        return [text.strip()]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_size
        
        if end < len(text):
            cut = text.rfind(". ", start, end)
            if cut != -1:
                end = cut + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks

# Préparer les chunks enrichis avec leurs métadonnées
def prepare_chunks(documents):
    all_chunks = []

    for doc in documents:
        # On enrichit le texte avec le titre et l'article
        # pour que l'embedding capture le contexte juridique
        enriched_text = (
            f"Article {doc['article']} — {doc['titre']}\n"
            f"Section : {doc['section']}\n"
            f"{doc['texte']}"
        )

        chunks = chunk_text(enriched_text)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "content": chunk,
                "metadata": {
                    "article": doc["article"],
                    "titre": doc["titre"],
                    "section": doc["section"],
                    "chunk_index": i
                }
            })

    print(f"Chunking done : {len(all_chunks)} chunks created")
    return all_chunks

# Charger le modèle d'embedding multilingue
def load_embedding_model():
    print("Loading embedding model...")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    print(f"Model loaded — vector dimension : {model.get_embedding_dimension()}")
    return model

# Transformer les chunks en vecteurs numpy
def embed_chunks(chunks, model):
    texts = [c["content"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    vectors = model.encode(texts, show_progress_bar=True)
    print(f"Embeddings done — shape : {vectors.shape}")
    return vectors.astype(np.float32)

# Créer l'index FAISS avec similarité cosinus (normalisation L2 + produit scalaire)
def create_faiss_index(vectors):
    dimension = vectors.shape[1]
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    print(f"FAISS index created — {index.ntotal} vectors indexed")
    return index

# Sauvegarder l'index FAISS et les métadonnées sur le disque
def save_index(index, chunks, folder):
    os.makedirs(folder, exist_ok=True)
    faiss.write_index(index, f"{folder}/faiss.index")
    with open(f"{folder}/chunks_meta.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Index saved in '{folder}/'")

# Point d'entrée principal — assemble toutes les étapes du pipeline d'indexation
def main():
    print("=" * 50)
    print("  PHASE 1 — INDEXATION")
    print("=" * 50)

    # Étape 1 : charger le corpus
    documents = load_corpus("corpus/code_travail.json")

    # Étape 2 : préparer les chunks
    chunks = prepare_chunks(documents)

    # Étape 3 : charger le modèle et créer les embeddings
    model = load_embedding_model()
    vectors = embed_chunks(chunks, model)

    # Étape 4 : créer et sauvegarder l'index FAISS
    index = create_faiss_index(vectors)
    save_index(index, chunks, "index")

    print("\n" + "=" * 50)
    print("  Indexation complete !")
    print("  You can now run : python rag.py")
    print("=" * 50)


if __name__ == "__main__":
    main()