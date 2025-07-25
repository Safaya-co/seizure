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


def classify_question_type(question):
    """Détermine si une question nécessite le RAG ou peut être répondue directement."""
    
    # Mots-clés indiquant une question médicale générale (pas besoin de RAG)
    general_keywords = [
        "bonjour", "salut", "hello", "comment allez-vous", "comment ça va",
        "qui êtes-vous", "que faites-vous", "pouvez-vous m'aider",
        "qu'est-ce que", "définition", "expliquer", "c'est quoi",
        "merci", "au revoir", "goodbye", "à bientôt"
    ]
    
    # Mots-clés indiquant une question spécialisée (besoin de RAG)
    specialized_keywords = [
        "crise", "épilepsie", "convulsion", "seizure", "eeg", "électroencéphalogramme",
        "diagnostic", "traitement", "médicament", "antiépileptique", "protocole",
        "symptôme", "signe", "manifestation", "patient", "cas clinique",
        "dosage", "posologie", "effet secondaire", "contre-indication"
    ]
    
    question_lower = question.lower()
    
    # Vérifier les mots-clés généraux
    general_score = sum(1 for keyword in general_keywords if keyword in question_lower)
    
    # Vérifier les mots-clés spécialisés
    specialized_score = sum(1 for keyword in specialized_keywords if keyword in question_lower)
    
    # Questions très courtes (moins de 10 mots) sont souvent générales
    word_count = len(question.split())
    
    # Logique de classification
    if general_score > 0 and specialized_score == 0:
        return "general", 0.9
    elif specialized_score > 0:
        return "specialized", min(0.9, 0.5 + (specialized_score * 0.1))
    elif word_count < 5:
        return "general", 0.7
    else:
        # Par défaut, traiter comme spécialisé mais avec moins de confiance
        return "specialized", 0.6


def get_conversation_context(current_question, max_context_items=5):
    """Récupère le contexte de conversation récent pour maintenir la mémoire."""
    history = load_history()
    
    if not history:
        return None, False
    
    # Prendre les N dernières interactions (exclure la question actuelle si elle est déjà sauvée)
    recent_history = history[-max_context_items:]
    
    # Vérifier s'il y a des références contextuelles dans la question actuelle
    contextual_indicators = [
        "comme vous avez dit", "comme mentionné", "en référence à", "suite à",
        "concernant votre réponse", "à propos de", "dans le cas précédent",
        "il", "elle", "cela", "ça", "ce", "cette", "celui", "celle",
        "et aussi", "également", "de plus", "en plus"
    ]
    
    has_contextual_reference = any(indicator in current_question.lower() 
                                 for indicator in contextual_indicators)
    
    # Construire le contexte de conversation
    if recent_history and (has_contextual_reference or len(recent_history) >= 2):
        context_parts = []
        for i, entry in enumerate(recent_history):
            context_parts.append(f"[Interaction {i+1}]")
            context_parts.append(f"Question: {entry['question']}")
            # Résumer la réponse si elle est longue
            answer = entry['answer']
            if len(answer) > 200:
                answer = answer[:200] + "..."
            context_parts.append(f"Réponse: {answer}")
            context_parts.append("")  # Ligne vide
        
        return "\n".join(context_parts), has_contextual_reference
    
    return None, has_contextual_reference


def build_prompt_with_memory(query, top_chunks, is_grounded=True, conversation_context=None):
    """Construit le prompt avec contexte de conversation si disponible."""
    
    # Prompt de base selon le type (grounded ou non)
    if is_grounded:
        base_instruction = (
            "Réponds uniquement avec les informations fournies dans les contextes ci-dessous. "
            "Si l'information n'est pas présente, dis 'Je ne sais pas'."
        )
    else:
        base_instruction = (
            "Les contextes suivants contiennent peu d'informations pertinentes pour cette question. "
            "Réponds en te basant sur tes connaissances médicales générales, tout en restant prudent et en recommandant une consultation médicale si nécessaire."
        )
    
    # Contexte documentaire
    if is_grounded:
        doc_context = "\n\n".join([f"[Doc {i+1}] {chunk.strip()}" for i, chunk in enumerate(top_chunks)])
    else:
        doc_context = "\n\n".join([f"[Doc {i+1}] {chunk.strip()}" for i, chunk in enumerate(top_chunks[:3])])
    
    # Construire le prompt avec ou sans mémoire conversationnelle
    if conversation_context:
        return f"""{base_instruction}

CONTEXTE DE CONVERSATION RÉCENTE :
{conversation_context}

CONTEXTES DOCUMENTAIRES :
{doc_context}

QUESTION ACTUELLE : {query}

Instructions supplémentaires :
- Tiens compte de l'historique de conversation pour donner une réponse cohérente
- Si la question fait référence à une discussion précédente, utilise ce contexte
- Reste dans le domaine médical et spécialisé en détection de crises

Réponse :"""
    else:
        return f"""{base_instruction}

CONTEXTES DOCUMENTAIRES :
{doc_context}

Question : {query}

Réponse :"""


def call_mistral_with_memory(prompt, is_grounded=True, has_memory_context=False):
    """Appel à Mistral avec gestion de la mémoire conversationnelle."""
    try:
        # Adapter le message système selon la présence de mémoire
        if has_memory_context:
            if is_grounded:
                system_message = (
                    "Tu es un assistant médical spécialisé en détection de crises. "
                    "Tu as accès à l'historique de la conversation et peux t'y référer pour donner des réponses cohérentes. "
                    "Utilise ce contexte pour personnaliser tes réponses."
                )
            else:
                system_message = (
                    "Tu es un assistant médical spécialisé en détection de crises. "
                    "Tu as accès à l'historique de la conversation. "
                    "IMPORTANT: Tu dois commencer ta réponse par '⚠️ ATTENTION: Cette réponse ne se base pas avec certitude sur la documentation fournie. ' "
                    "puis donner une réponse basée sur tes connaissances générales et le contexte conversationnel."
                )
        else:
            # Messages système originaux si pas de contexte mémoire
            if is_grounded:
                system_message = "Tu es un assistant médical spécialisé en détection de crises."
            else:
                system_message = (
                    "Tu es un assistant médical spécialisé en détection de crises. "
                    "IMPORTANT: Tu dois commencer ta réponse par '⚠️ ATTENTION: Cette réponse ne se base pas avec certitude sur la documentation fournie. ' "
                    "puis donner une réponse basée sur tes connaissances générales tout en restant prudent."
                )
        
        response = client.chat.completions.create(
            model="mistral-medium-2505",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Erreur API Mistral (avec mémoire): {e}")
        return f"[ERREUR] {e}"


def call_mistral_without_rag_with_memory(question, conversation_context=None):
    """Répond à une question générale avec mémoire conversationnelle."""
    try:
        # Construire le prompt avec contexte si disponible
        if conversation_context:
            prompt = f"""CONTEXTE DE CONVERSATION RÉCENTE :
{conversation_context}

QUESTION ACTUELLE : {question}

Instructions :
- Tiens compte de l'historique pour donner une réponse cohérente
- Réponds de manière concise et professionnelle
- Pour les questions médicales spécifiques, recommande une consultation médicale

Réponse :"""
            system_message = (
                "Tu es un assistant médical spécialisé en détection de crises. "
                "Tu as accès à l'historique de conversation et peux t'y référer pour maintenir la cohérence."
            )
        else:
            prompt = question
            system_message = (
                "Tu es un assistant médical spécialisé en détection de crises. "
                "Réponds de manière concise et professionnelle. "
                "Pour les questions médicales spécifiques, recommande toujours une consultation médicale."
            )
        
        response = client.chat.completions.create(
            model="mistral-medium-2505",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=256,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Erreur API Mistral (sans RAG avec mémoire): {e}")
        return f"[ERREUR] {e}"


def generate_short_summary(long_answer):
    """Génère un résumé court d'une réponse longue."""
    # Seuil pour considérer une réponse comme longue
    if len(long_answer.split()) < 50:
        return None  # Pas besoin de résumé pour les réponses courtes
    
    try:
        response = client.chat.completions.create(
            model="mistral-medium-2505",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu dois résumer le texte médical fourni en 2-3 phrases maximum. "
                        "Garde les informations essentielles et les recommandations importantes. "
                        "Commence par 'En résumé:'"
                    )
                },
                {
                    "role": "user",
                    "content": f"Résume ce texte médical:\n\n{long_answer}"
                },
            ],
            temperature=0.2,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Erreur génération résumé: {e}")
        return None


def compute_similarity(answer, context):
    """Calcule la similarité cosinus entre réponse et contexte."""
    emb_ans = embedding_model.encode(answer, convert_to_tensor=True, normalize_embeddings=True)
    emb_ctx = embedding_model.encode(context, convert_to_tensor=True, normalize_embeddings=True)
    return util.cos_sim(emb_ans, emb_ctx).item()


def save_to_history(question, answer, score, grounded, question_type, used_rag, memory_used):
    """Sauvegarde l'interaction dans le fichier JSON d'historique."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "cosine_score": round(score, 3),
        "grounded": grounded,
        "confidence_level": "haute" if grounded else "faible",
        "has_warning": answer.startswith("⚠️"),
        "question_type": question_type,
        "used_rag": used_rag,
        "memory_used": memory_used,
        "answer_length": len(answer.split())
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

    # ÉTAPE 1: Classification de la question
    question_type, classification_confidence = classify_question_type(question)
    print(f"📝 Question classifiée: {question_type} (confiance: {classification_confidence})")
    
    # ÉTAPE 2: Récupération du contexte conversationnel
    conversation_context, has_contextual_ref = get_conversation_context(question)
    memory_used = conversation_context is not None
    print(f"🧠 Mémoire: {'Utilisée' if memory_used else 'Non utilisée'} | Référence contextuelle: {has_contextual_ref}")
    
    # ÉTAPE 3: Traitement selon le type de question
    if question_type == "general":
        # Réponse directe sans RAG mais avec mémoire possible
        answer = call_mistral_without_rag_with_memory(question, conversation_context)
        cosine_score = 0.0  # Pas de score RAG pour les questions générales
        grounded = False
        used_rag = False
        short_summary = None
        
    else:
        # Utiliser le RAG pour les questions spécialisées avec mémoire
        used_rag = True
        top_chunks = get_top_k_chunks(question)
        context_concat = " ".join(top_chunks)
        
        # Calculer la similarité AVANT de générer la réponse
        temp_similarity = compute_similarity(question, context_concat)
        grounded = temp_similarity >= 0.4
        
        # Construire le prompt adapté avec mémoire
        prompt = build_prompt_with_memory(question, top_chunks, is_grounded=grounded, conversation_context=conversation_context)
        answer = call_mistral_with_memory(prompt, is_grounded=grounded, has_memory_context=memory_used)
        
        # Calculer la vraie similarité avec la réponse finale
        cosine_score = compute_similarity(answer, context_concat)
        final_grounded = cosine_score >= 0.7
        
        # Ajuster le statut si nécessaire
        if not grounded and not answer.startswith("⚠️"):
            answer = f"⚠️ ATTENTION: Cette réponse ne se base pas avec certitude sur la documentation fournie.\n\n{answer}"
        
        grounded = final_grounded
        
        # ÉTAPE 4: Générer un résumé court si la réponse est longue
        short_summary = generate_short_summary(answer)
    
    # Sauvegarde dans l'historique
    save_to_history(question, answer, cosine_score, grounded, question_type, used_rag, memory_used)
    
    # Préparer la réponse
    response_data = {
        "response": answer,
        "cosine_score": round(cosine_score, 3),
        "grounded_in_docs": grounded,
        "confidence_level": "haute" if grounded else "faible",
        "question_type": question_type,
        "classification_confidence": round(classification_confidence, 2),
        "used_rag": used_rag,
        "memory_used": memory_used,
        "has_contextual_reference": has_contextual_ref,
        "warning_displayed": answer.startswith("⚠️")
    }
    
    # Ajouter le résumé court s'il existe
    if short_summary:
        response_data["short_summary"] = short_summary
        response_data["has_short_summary"] = True
    else:
        response_data["has_short_summary"] = False
    
    return jsonify(response_data)

@app.route("/history", methods=["GET"])
def history():
    return jsonify(load_history())

# --- Lancement de l'application ---
if __name__ == "__main__":
    app.run(debug=True, port=5001)