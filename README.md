# Assistant Code du Travail — RAG

## Description
Un agent qui répond à des questions sur le Code du travail français en utilisant le système RAG (Retrieval-Augmented Generation). Il recherche les articles pertinents dans une base vectorielle FAISS et génère une réponse via le LLM Groq.

## Installation

### 1. Cloner le repo
git clone https://github.com/LegreArnold/rag_code_travail.git
cd rag_code_travail

### 2. Créer et activer l'environnement virtuel
python -m venv venv
venv\Scripts\activate

### 3. Installer les dépendances
pip install -r requirements.txt

### 4. Lancer l'indexation 
python indexation.py

### 5. Lancer l'assistant
python rag.py

## Structure du projet
rag_code_travail/
├── corpus/
│   └── code_travail.json     # Articles du Code du travail
├── index/
│   ├── faiss.index           # Index vectoriel (généré par indexation.py)
│   └── chunks_meta.json      # Métadonnées des chunks
├── indexation.py             # Phase 1 : indexation du corpus
├── rag.py                    # Phase 2 : questions-réponses
├── requirements.txt
└── .env                      # Clé API 

## Thèmes couverts
1. Durée du travail et heures supplémentaires
2. Congés payés
3. Contrat de travail (CDI, CDD)
4. Licenciement
5. Rupture conventionnelle

## Choix techniques

### Corpus manuel
On a choisi l'option corpus JSON manuel avec 22 articles officiels
issus de Légifrance. Ce choix permet un contrôle total sur la qualité
des données et évite la complexité des API externes.

### Chunking
Taille de chunk : 500 caractères avec overlap de 50 caractères.
Le découpage privilégie les fins de phrases pour ne pas couper
une règle juridique en plein milieu.

### Modèle d'embedding
Modèle : paraphrase-multilingual-mpnet-base-v2
Choisi car le corpus est en français. Ce modèle supporte
plusieurs langues dont le français.

### Index FAISS — IndexFlatIP
Après normalisation L2 des vecteurs, le produit scalaire est
équivalent à la similarité cosinus. Un score proche de 1.0
signifie une grande similarité sémantique.

### LLM — llama-3.3-70b-versatile via Groq
Modèle puissant pour la précision juridique. Temperature fixée
à 0.1 pour des réponses factuelles et minimiser les hallucinations.

## Bonus implémenté
Score de confiance : si le meilleur score est inférieur à 0.45,
le système affiche un avertissement avant la réponse.
