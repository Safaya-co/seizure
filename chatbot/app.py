# app.py - Backend Flask avec historique JSON pour assistant médical

from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from dotenv import load_dotenv
from openai import OpenAI
import torch
import pickle
import os
import json
from datetime import datetime

# --- Charger variables d'environnement depuis fichier .env ---
load_dotenv()

# --- Configuration Flask ---
app = Flask(__name__)

# --- Chemins absolus ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RETRIEVAL_FILE = os.path.join(BASE_DIR, "data", "retrieval_data_bge_v15.pkl")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "data", "embeddings_bge_v15.pt")
HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

# --- Configuration API Mistral via OpenAI SDK ---
api_key = os.getenv("MISTRAL_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Aucun API key trouvé. Définir MISTRAL_API_KEY ou OPENAI_API_KEY dans .env.")

# CORRECTION: Configuration correcte pour Mistral
client = OpenAI(
    api_key=api_key,
    base_url="https://api.mistral.ai/v1"  # Utiliser base_url au lieu de api_base
)
print(f"🔑 Clé API chargée (masquée): {api_key[:6]}***{api_key[-4:]}")

# --- Test de connexion à l'API Mistral ---
try:
    models = client.models.list()
    count = len(models.data) if hasattr(models, 'data') else len(models)
    print(f"✅ Connexion à l'API Mistral réussie, {count} modèles disponibles.")
    # Afficher les modèles disponibles
    if hasattr(models, 'data'):
        print("Modèles disponibles:")
        for model in models.data[:5]:  # Afficher les 5 premiers
            print(f"  - {model.id}")
except Exception as e:
    print(f"❌ Échec de la connexion à l'API Mistral: {e}")

# --- Chargement des modèles locaux ---
print("Loading embedding model...")
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
print("Loading reranker model...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Chargement des données de retrieval ---
with open(RETRIEVAL_FILE, "rb") as f:
    retrieval_data = pickle.load(f)
all_chunks = retrieval_data.get("chunk", retrieval_data.get("chunks", [])).fillna("").astype(str).tolist()
chunk_embeddings = torch.load(EMBEDDINGS_FILE, map_location="cpu")

# --- Fonctions auxiliaires ---
def get_top_k_chunks(query, k=20, rerank_k=10):
    """Retourne les top-k chunks rerankés pour une question."""
    q_emb = embedding_model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(q_emb, chunk_embeddings)[0]
    top_idx = torch.topk(scores, k=k).indices.tolist()
    candidates = [all_chunks[i] for i in top_idx]
    rerank_inputs = [(query, c) for c in candidates]
    rerank_scores = reranker.predict(rerank_inputs)
    ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
    return [text for text, _ in ranked[:rerank_k]]


def build_prompt(query, top_chunks):
    """Construit le prompt pour Mistral avec contexte."""
    instruction = (
        "Réponds uniquement avec les informations fournies dans les contextes ci-dessous. "
        "Si l'information n'est pas présente, dis 'Je ne sais pas'."
    )
    context = "\n\n".join([f"[{i+1}] {chunk.strip()}" for i, chunk in enumerate(top_chunks)])
    return f"""{instruction}

Contextes :
{context}

Question : {query}

Réponse :"""


def call_mistral_api(prompt):
    """Envoie un prompt à l'API Mistral et retourne la réponse."""
    try:
        response = client.chat.completions.create(
            model="mistral-medium-2505",  # Modèle disponible selon votre test
            messages=[
                {"role": "system", "content": "Tu es un assistant médical spécialisé en détection de crises."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Erreur API Mistral: {e}")
        return f"[ERREUR] {e}"


def compute_similarity(answer, context):
    """Calcule la similarité cosinus entre réponse et contexte."""
    emb_ans = embedding_model.encode(answer, convert_to_tensor=True, normalize_embeddings=True)
    emb_ctx = embedding_model.encode(context, convert_to_tensor=True, normalize_embeddings=True)
    return util.cos_sim(emb_ans, emb_ctx).item()


def save_to_history(question, answer, score, grounded):
    """Sauvegarde l'interaction dans le fichier JSON d'historique."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "cosine_score": round(score, 3),
        "grounded": grounded
    }
    history = load_history()
    history.append(entry)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def load_history():
    """Charge l'historique JSON si existant, sinon renvoie []"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

# --- Routes Flask ---
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    question = data.get("message", "").strip()
    if not question:
        return jsonify({"response": "Aucune question fournie."})

    top_chunks = get_top_k_chunks(question)
    prompt = build_prompt(question, top_chunks)
    answer = call_mistral_api(prompt)
    context_concat = " ".join(top_chunks)
    cosine_score = compute_similarity(answer, context_concat)
    grounded = cosine_score >= 0.7

    save_to_history(question, answer, cosine_score, grounded)
    return jsonify({
        "response": answer,
        "cosine_score": round(cosine_score, 3),
        "grounded_in_docs": grounded
    })

@app.route("/history", methods=["GET"])
def history():
    return jsonify(load_history())

# --- Lancement de l'application ---
if __name__ == "__main__":
    app.run(debug=True, port=5001)