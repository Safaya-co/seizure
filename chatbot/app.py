# app.py - Backend Flask avec historique JSON pour assistant médical (Version avec gestion de sessions)

from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from dotenv import load_dotenv
from openai import OpenAI
import torch
import pickle
import os
import json
import uuid
import shutil
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

# --- Gestion des sessions ---
current_session_id = str(uuid.uuid4())
print(f"🆔 Session actuelle: {current_session_id[:8]}...")

# --- Configuration API Mistral via OpenAI SDK ---
api_key = os.getenv("MISTRAL_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Aucun API key trouvé. Définir MISTRAL_API_KEY ou OPENAI_API_KEY dans .env.")

# Configuration correcte pour Mistral
client = OpenAI(
    api_key=api_key,
    base_url="https://api.mistral.ai/v1"
)
print(f"🔑 Clé API chargée (masquée): {api_key[:6]}***{api_key[-4:]}")

# --- Test de connexion à l'API Mistral ---
try:
    models = client.models.list()
    count = len(models.data) if hasattr(models, 'data') else len(models)
    print(f"✅ Connexion à l'API Mistral réussie, {count} modèles disponibles.")
    if hasattr(models, 'data'):
        print("Modèles disponibles:")
        for model in models.data[:5]:
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

# --- Fonctions de gestion de l'historique ---
def load_history():
    """Charge l'historique JSON si existant, sinon renvoie [] - VERSION ULTRA ROBUSTE"""
    try:
        if os.path.exists(HISTORY_FILE):
            file_size = os.path.getsize(HISTORY_FILE)
            if file_size == 0:
                print("⚠️ Fichier historique vide détecté, initialisation automatique...")
                _initialize_empty_history_file()
                return []
            
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                if not content:
                    print("⚠️ Contenu historique vide après nettoyage, initialisation...")
                    _initialize_empty_history_file()
                    return []
                
                if len(content) < 2:
                    print("⚠️ Contenu historique insuffisant, initialisation...")
                    _initialize_empty_history_file()
                    return []
                
                try:
                    history = json.loads(content)
                except json.JSONDecodeError as json_err:
                    print(f"❌ Fichier JSON malformé: {json_err}")
                    print(f"📋 Contenu du fichier (premiers 100 chars): '{content[:100]}'")
                    _backup_and_reinitialize_history()
                    return []
                
                if not isinstance(history, list):
                    print(f"⚠️ Format historique invalide (type: {type(history)}), initialisation...")
                    _backup_and_reinitialize_history()
                    return []
                
                print(f"✅ Historique chargé avec succès: {len(history)} entrées")
                return history
        else:
            print("📝 Aucun fichier historique trouvé, création d'un nouveau fichier...")
            _initialize_empty_history_file()
            return []
            
    except PermissionError as e:
        print(f"❌ Problème de permissions sur le fichier historique: {e}")
        return []
        
    except Exception as e:
        print(f"❌ Erreur inattendue lors du chargement de l'historique: {e}")
        print(f"📋 Type d'erreur: {type(e).__name__}")
        return []


def _initialize_empty_history_file():
    """Initialise un fichier d'historique vide avec un JSON valide."""
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        
        print(f"✅ Fichier d'historique initialisé: {HISTORY_FILE}")
        
    except Exception as e:
        print(f"❌ Impossible d'initialiser le fichier d'historique: {e}")


def _backup_and_reinitialize_history():
    """Sauvegarde un fichier corrompu et en crée un nouveau."""
    try:
        if os.path.exists(HISTORY_FILE):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = HISTORY_FILE + f".corrupted.{timestamp}"
            
            try:
                shutil.copy2(HISTORY_FILE, backup_file)
                print(f"💾 Fichier corrompu sauvegardé: {backup_file}")
            except Exception as backup_error:
                print(f"❌ Impossible de sauvegarder le fichier corrompu: {backup_error}")
        
        _initialize_empty_history_file()
        
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde et réinitialisation: {e}")


def initialize_history_file():
    """Initialise le fichier d'historique s'il n'existe pas ou est corrompu."""
    try:
        history = load_history()
        
        if not history and not os.path.exists(HISTORY_FILE):
            os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2, ensure_ascii=False)
            print("📝 Fichier d'historique initialisé")
            
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation de l'historique: {e}")


# --- Fonctions RAG et IA ---
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
    
    general_keywords = [
        "bonjour", "salut", "hello", "comment allez-vous", "comment ça va",
        "qui êtes-vous", "que faites-vous", "pouvez-vous m'aider",
        "qu'est-ce que", "définition", "expliquer", "c'est quoi",
        "merci", "au revoir", "goodbye", "à bientôt"
    ]
    
    specialized_keywords = [
        "crise", "épilepsie", "convulsion", "seizure", "eeg", "électroencéphalogramme",
        "diagnostic", "traitement", "médicament", "antiépileptique", "protocole",
        "symptôme", "signe", "manifestation", "patient", "cas clinique",
        "dosage", "posologie", "effet secondaire", "contre-indication"
    ]
    
    question_lower = question.lower()
    
    general_score = sum(1 for keyword in general_keywords if keyword in question_lower)
    specialized_score = sum(1 for keyword in specialized_keywords if keyword in question_lower)
    
    word_count = len(question.split())
    
    if general_score > 0 and specialized_score == 0:
        return "general", 0.9
    elif specialized_score > 0:
        return "specialized", min(0.9, 0.5 + (specialized_score * 0.1))
    elif word_count < 5:
        return "general", 0.7
    else:
        return "specialized", 0.6


def get_conversation_context(current_question, max_context_items=5):
    """Récupère le contexte de conversation récent pour la session actuelle."""
    global current_session_id
    
    history = load_history()
    
    if not history:
        return None, False
    
    # Filtrer par session actuelle
    session_history = [entry for entry in history if entry.get('session_id') == current_session_id]
    
    if not session_history:
        return None, False
    
    recent_history = session_history[-max_context_items:]
    
    contextual_indicators = [
        "comme vous avez dit", "comme mentionné", "en référence à", "suite à",
        "concernant votre réponse", "à propos de", "dans le cas précédent",
        "il", "elle", "cela", "ça", "ce", "cette", "celui", "celle",
        "et aussi", "également", "de plus", "en plus"
    ]
    
    has_contextual_reference = any(indicator in current_question.lower() 
                                 for indicator in contextual_indicators)
    
    if recent_history and (has_contextual_reference or len(recent_history) >= 2):
        context_parts = []
        for i, entry in enumerate(recent_history):
            context_parts.append(f"[Interaction {i+1}]")
            context_parts.append(f"Question: {entry['question']}")
            answer = entry['answer']
            if len(answer) > 200:
                answer = answer[:200] + "..."
            context_parts.append(f"Réponse: {answer}")
            context_parts.append("")
        
        return "\n".join(context_parts), has_contextual_reference
    
    return None, has_contextual_reference


def build_prompt_with_memory(query, top_chunks, is_grounded=True, conversation_context=None):
    """Construit le prompt avec contexte de conversation si disponible."""
    
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
    
    if is_grounded:
        doc_context = "\n\n".join([f"[Doc {i+1}] {chunk.strip()}" for i, chunk in enumerate(top_chunks)])
    else:
        doc_context = "\n\n".join([f"[Doc {i+1}] {chunk.strip()}" for i, chunk in enumerate(top_chunks[:3])])
    
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
    if len(long_answer.split()) < 50:
        return None
    
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



def generate_bullet_points(long_answer):
    """Génère une version en points clés d'une réponse longue."""
    if len(long_answer.split()) < 30:
        return None
    
    try:
        response = client.chat.completions.create(
            model="mistral-medium-2505",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu dois transformer le texte médical fourni en points clés avec des tirets. "
                        "Présente les informations sous forme de liste structurée avec des tirets (-). "
                        "Garde les informations essentielles et les recommandations importantes. "
                        "Maximum 5-7 points. "
                        "Format: - Point 1\n- Point 2\n- etc."
                    )
                },
                {
                    "role": "user",
                    "content": f"Transforme ce texte médical en points clés avec des tirets:\n\n{long_answer}"
                },
            ],
            temperature=0.2,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Erreur génération points clés: {e}")
        return None



def compute_similarity(answer, context):
    """Calcule la similarité cosinus entre réponse et contexte."""
    emb_ans = embedding_model.encode(answer, convert_to_tensor=True, normalize_embeddings=True)
    emb_ctx = embedding_model.encode(context, convert_to_tensor=True, normalize_embeddings=True)
    return util.cos_sim(emb_ans, emb_ctx).item()


def save_to_history(question, answer, score, grounded, question_type, used_rag, memory_used, short_summary=None, bullet_points=None):
    """Sauvegarde l'interaction dans le fichier JSON d'historique avec session."""
    global current_session_id
    
    entry = {
        "session_id": current_session_id,
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
        "answer_length": len(answer.split()),
        "short_summary": short_summary,
        "bullet_points": bullet_points
    }
    
    try:
        history = load_history()
        history.append(entry)
        
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Historique sauvegardé: {len(history)} entrées (Session: {current_session_id[:8]})")
        
    except Exception as e:
        print(f"❌ Erreur sauvegarde historique: {e}")



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

    # Initialisation des variables pour éviter UnboundLocalError
    bullet_points = None
    short_summary = None

    # ÉTAPE 1: Classification de la question
    question_type, classification_confidence = classify_question_type(question)
    print(f"📝 Question classifiée: {question_type} (confiance: {classification_confidence})")
    
    # ÉTAPE 2: Récupération du contexte conversationnel
    conversation_context, has_contextual_ref = get_conversation_context(question)
    memory_used = conversation_context is not None
    print(f"🧠 Mémoire: {'Utilisée' if memory_used else 'Non utilisée'} | Référence contextuelle: {has_contextual_ref}")
    
    # ÉTAPE 3: Traitement selon le type de question
    if question_type == "general":
        answer = call_mistral_without_rag_with_memory(question, conversation_context)
        cosine_score = 0.0
        grounded = False
        used_rag = False
        # short_summary et bullet_points restent None (déjà initialisés)
        
    else:
        used_rag = True
        top_chunks = get_top_k_chunks(question)
        context_concat = " ".join(top_chunks)
        
        temp_similarity = compute_similarity(question, context_concat)
        grounded = temp_similarity >= 0.4
        
        prompt = build_prompt_with_memory(question, top_chunks, is_grounded=grounded, conversation_context=conversation_context)
        answer = call_mistral_with_memory(prompt, is_grounded=grounded, has_memory_context=memory_used)
        
        cosine_score = compute_similarity(answer, context_concat)
        final_grounded = cosine_score >= 0.7
        
        if not grounded and not answer.startswith("⚠️"):
            answer = f"⚠️ ATTENTION: Cette réponse ne se base pas avec certitude sur la documentation fournie.\n\n{answer}"
        
        grounded = final_grounded
        short_summary = generate_short_summary(answer)
        bullet_points = generate_bullet_points(answer)
    
    # Sauvegarde dans l'historique
    save_to_history(question, answer, cosine_score, grounded, question_type, used_rag, memory_used, short_summary, bullet_points)
    
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
    
    # Ajout du résumé si disponible
    if short_summary:
        response_data["short_summary"] = short_summary
        response_data["has_short_summary"] = True
    else:
        response_data["has_short_summary"] = False
    
    # Ajout des points clés si disponibles
    if bullet_points:
        response_data["bullet_points"] = bullet_points
        response_data["has_bullet_points"] = True
    else:
        response_data["has_bullet_points"] = False
    
    return jsonify(response_data)



@app.route("/history", methods=["GET"])
def history():
    return jsonify(load_history())


@app.route("/session-history", methods=["GET"])
def session_history():
    """Retourne l'historique de la session actuelle."""
    global current_session_id
    
    history = load_history()
    session_history = [entry for entry in history if entry.get('session_id') == current_session_id]
    
    return jsonify({
        "session_id": current_session_id,
        "messages": session_history
    })


@app.route("/new-session", methods=["POST"])
def new_session():
    """Démarre une nouvelle session."""
    global current_session_id
    current_session_id = str(uuid.uuid4())
    print(f"🆔 Nouvelle session créée: {current_session_id[:8]}...")
    
    return jsonify({
        "session_id": current_session_id,
        "message": "Nouvelle session créée"
    })


@app.route("/reset-history", methods=["POST"])
def reset_history():
    """Route pour réinitialiser l'historique en cas de problème."""
    try:
        if os.path.exists(HISTORY_FILE):
            backup_file = HISTORY_FILE + f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(HISTORY_FILE, backup_file)
            print(f"💾 Historique sauvegardé: {backup_file}")
        
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        
        return jsonify({"status": "success", "message": "Historique réinitialisé avec succès"})
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"Erreur lors de la réinitialization: {e}"})


# --- Lancement de l'application ---
if __name__ == "__main__":
    initialize_history_file()
    app.run(debug=True, port=5001)
