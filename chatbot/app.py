# chatbot_app.py - Backend Flask pour un assistant médical basé RAG (CPU uniquement)

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import torch
import pickle
import os

# --- Initialisation de l'application Flask ---
app = Flask(__name__)

# --- Chargement des ressources ---
# Modèle d'encodage BGE pour les questions et documents
embedding_model_name = "BAAI/bge-base-en-v1.5"
embedding_model = SentenceTransformer(embedding_model_name, device="cpu")

# Modèle CrossEncoder pour le reranking (plus lent mais précis)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Chargement des données de chunks indexés
with open("retrieval_data_bge_v15.pkl", "rb") as f:
    retrieval_data = pickle.load(f)

# Préparation des données de recherche
all_chunks = retrieval_data["chunk"].fillna("").astype(str).tolist()
chunk_embeddings = torch.load("embeddings_bge_v15.pt", map_location=torch.device("cpu"))

# --- Fonctions auxiliaires ---
def get_top_k_chunks(query, k=20, rerank_k=10):
    """Retourne les top-k chunks rerankés par CrossEncoder pour une requête."""
    query_embedding = embedding_model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    top_indices = torch.topk(scores, k=k).indices.tolist()

    candidates = [all_chunks[i] for i in top_indices]
    rerank_inputs = [(query, chunk) for chunk in candidates]
    rerank_scores = reranker.predict(rerank_inputs)

    reranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
    return [text for text, score in reranked[:rerank_k]]

def build_prompt(query, top_chunks):
    """Construit un prompt contextuel pour l'API Mistral."""
    context = "\n\n".join([f"[{i+1}] {chunk.strip()}" for i, chunk in enumerate(top_chunks)])
    instruction = (
        "Réponds uniquement avec les informations fournies dans les contextes ci-dessous. "
        "Si l'information n'est pas présente, dis 'Je ne sais pas'."
    )
    return f"""{instruction}

Contextes :
{context}

Question : {query}

Réponse :"""

def call_mistral_api(prompt):
    """Envoie un prompt à l'API Mistral et retourne la réponse générée."""
    import openai
    openai.api_key = os.getenv("MISTRAL_API_KEY")
    openai.api_base = "https://api.mistral.ai/v1"

    try:
        response = openai.ChatCompletion.create(
            model="mistral-small",
            messages=[
                {"role": "system", "content": "Tu es un assistant médical spécialisé en détection de crises."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=512,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"[ERREUR] {e}"

def compute_similarity_metrics(answer, top_chunks_text):
    """Calcule la similarité cosine entre la réponse et le contexte concaténé."""
    emb_answer = embedding_model.encode(answer, convert_to_tensor=True, normalize_embeddings=True)
    emb_chunks = embedding_model.encode(top_chunks_text, convert_to_tensor=True, normalize_embeddings=True)
    cosine_sim = util.cos_sim(emb_answer, emb_chunks).item()
    return cosine_sim

@app.route("/")
def index():
    return render_template("index.html")

# --- API principale ---
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("message", "")
    if not question:
        return jsonify({"response": "Aucune question fournie."})

    top_chunks = get_top_k_chunks(question)
    prompt = build_prompt(question, top_chunks)
    answer = call_mistral_api(prompt)

    top_chunks_concat = " ".join(top_chunks)
    cosine_score = compute_similarity_metrics(answer, top_chunks_concat)
    grounded = cosine_score >= 0.7

    return jsonify({
        "response": answer,
        "cosine_score": round(cosine_score, 3),
        "grounded_in_docs": grounded
    })

# --- Lancement local ---
if __name__ == "__main__":
    app.run(debug=True, port=5000)