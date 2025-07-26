# app.py - Backend Flask avec système multi-conversations
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
import glob
from datetime import datetime

# --- Charger variables d'environnement depuis fichier .env ---
load_dotenv()

# --- Configuration Flask ---
app = Flask(__name__)

# --- Chemins absolus ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RETRIEVAL_FILE = os.path.join(BASE_DIR, "data", "retrieval_data_bge_v15.pkl")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "data", "embeddings_bge_v15.pt")
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- Gestion des sessions ---
current_session_id = str(uuid.uuid4())
print(f"🆔 Session actuelle: {current_session_id[:8]}...")

# --- Configuration API Mistral via OpenAI SDK ---
api_key = os.getenv("MISTRAL_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Aucun API key trouvé. Définir MISTRAL_API_KEY ou OPENAI_API_KEY dans .env.")

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

# --- Fonctions de gestion multi-conversations ---

def get_conversation_file_path(session_id):
    """Retourne le chemin du fichier de conversation pour une session donnée."""
    return os.path.join(DATA_DIR, f"chat_history_{session_id}.json")

def create_new_conversation():
    """Crée une nouvelle conversation avec un ID unique."""
    new_session_id = str(uuid.uuid4())
    conversation_file = get_conversation_file_path(new_session_id)
    
    # Créer le dossier data s'il n'existe pas
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Initialiser le fichier de conversation
    initial_data = {
        "session_id": new_session_id,
        "created_at": datetime.now().isoformat(),
        "title": "Nouvelle conversation",
        "messages": []
    }
    
    try:
        with open(conversation_file, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Nouvelle conversation créée: {new_session_id[:8]}...")
        return new_session_id
        
    except Exception as e:
        print(f"❌ Erreur création conversation: {e}")
        return None

def load_conversation(session_id):
    """Charge une conversation spécifique par son ID."""
    conversation_file = get_conversation_file_path(session_id)
    
    try:
        if os.path.exists(conversation_file):
            with open(conversation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("messages", []), data
        else:
            print(f"⚠️ Fichier de conversation non trouvé: {session_id[:8]}")
            return [], None
            
    except Exception as e:
        print(f"❌ Erreur chargement conversation {session_id[:8]}: {e}")
        return [], None

def save_to_conversation(session_id, question, answer, score, grounded, question_type, used_rag, memory_used, short_summary=None, bullet_points=None):
    """Sauvegarde une interaction dans la conversation spécifiée."""
    conversation_file = get_conversation_file_path(session_id)
    
    # Charger la conversation existante
    try:
        if os.path.exists(conversation_file):
            with open(conversation_file, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
        else:
            # Créer une nouvelle conversation si elle n'existe pas
            conversation_data = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "title": "Nouvelle conversation",
                "messages": []
            }
    except Exception as e:
        print(f"❌ Erreur chargement pour sauvegarde: {e}")
        return False
    
    # Créer l'entrée
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
        "answer_length": len(answer.split()),
        "short_summary": short_summary,
        "bullet_points": bullet_points
    }
    
    # Ajouter à la liste des messages
    conversation_data["messages"].append(entry)
    
    # Mise à jour du titre si c'est le premier message
    if len(conversation_data["messages"]) == 1:
        # Générer un titre basé sur la première question
        title = question if len(question) <= 50 else question[:47] + "..."
        conversation_data["title"] = title
        conversation_data["updated_at"] = datetime.now().isoformat()
    else:
        conversation_data["updated_at"] = datetime.now().isoformat()
    
    # Sauvegarder
    try:
        with open(conversation_file, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Conversation sauvegardée: {len(conversation_data['messages'])} messages (Session: {session_id[:8]})")
        return True
        
    except Exception as e:
        print(f"❌ Erreur sauvegarde conversation: {e}")
        return False

def get_all_conversations():
    """Retourne la liste de toutes les conversations avec leurs métadonnées."""
    conversations = []
    
    try:
        # Chercher tous les fichiers chat_history_*.json dans le dossier data
        pattern = os.path.join(DATA_DIR, "chat_history_*.json")
        conversation_files = glob.glob(pattern)
        
        for file_path in conversation_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Extraire les métadonnées
                    session_id = data.get("session_id")
                    title = data.get("title", "Conversation sans titre")
                    created_at = data.get("created_at")
                    updated_at = data.get("updated_at", created_at)
                    message_count = len(data.get("messages", []))
                    
                    # Obtenir le dernier message pour aperçu
                    last_message = ""
                    if data.get("messages"):
                        last_msg = data["messages"][-1]
                        last_message = last_msg.get("question", "")[:100]
                        if len(last_message) >= 100:
                            last_message += "..."
                    
                    conversations.append({
                        "session_id": session_id,
                        "title": title,
                        "created_at": created_at,
                        "updated_at": updated_at,
                        "message_count": message_count,
                        "last_message": last_message,
                        "file_path": file_path
                    })
                    
            except Exception as e:
                print(f"❌ Erreur lecture fichier {file_path}: {e}")
                continue
        
        # Trier par date de mise à jour (plus récent en premier)
        conversations.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        
        return conversations
        
    except Exception as e:
        print(f"❌ Erreur récupération conversations: {e}")
        return []

def get_conversation_context(current_question, session_id, max_context_items=5):
    """Récupère le contexte de conversation récent pour une session spécifique."""
    messages, _ = load_conversation(session_id)
    
    if not messages:
        return None, False
    
    recent_history = messages[-max_context_items:]
    
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

# --- Fonctions RAG et IA (gardées identiques) ---
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
                "Tu has accès à l'historique de conversation et peux t'y référer pour maintenir la cohérence."
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


@app.route("/generate-summary", methods=["POST"])
def generate_summary_endpoint():
    """Route pour générer un résumé d'un texte."""
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    
    if not text:
        return jsonify({"error": "Texte manquant"}), 400
    
    try:
        summary = generate_short_summary(text)
        if summary:
            return jsonify({"summary": summary})
        else:
            return jsonify({"summary": "Résumé non disponible pour ce texte."})
    except Exception as e:
        print(f"Erreur génération résumé: {e}")
        return jsonify({"error": "Erreur lors de la génération du résumé"}), 500

@app.route("/generate-bullet-points", methods=["POST"])
def generate_bullet_points_endpoint():
    """Route pour générer des points clés d'un texte."""
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    
    if not text:
        return jsonify({"error": "Texte manquant"}), 400
    
    try:
        bullet_points = generate_bullet_points(text)
        if bullet_points:
            # Convertir le texte en liste de points
            points_list = [line.strip('- ').strip() for line in bullet_points.split('\n') if line.strip() and line.strip().startswith('-')]
            return jsonify({"bullet_points": points_list})
        else:
            return jsonify({"bullet_points": ["Points clés non disponibles pour ce texte."]})
    except Exception as e:
        print(f"Erreur génération points clés: {e}")
        return jsonify({"error": "Erreur lors de la génération des points clés"}), 500


def compute_similarity(answer, context):
    """Calcule la similarité cosinus entre réponse et contexte."""
    emb_ans = embedding_model.encode(answer, convert_to_tensor=True, normalize_embeddings=True)
    emb_ctx = embedding_model.encode(context, convert_to_tensor=True, normalize_embeddings=True)
    return util.cos_sim(emb_ans, emb_ctx).item()

# --- Routes Flask ---

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global current_session_id
    
    data = request.get_json() or {}
    question = data.get("message", "").strip()
    session_id = data.get("session_id") or current_session_id
    
    # Mettre à jour la session courante si nécessaire
    if session_id != current_session_id:
        current_session_id = session_id
    
    if not question:
        return jsonify({"error": "Aucune question fournie."})

    # Initialisation des variables
    bullet_points = None
    short_summary = None

    # ÉTAPE 1: Classification de la question
    question_type, classification_confidence = classify_question_type(question)
    print(f"📝 Question classifiée: {question_type} (confiance: {classification_confidence})")
    
    # ÉTAPE 2: Récupération du contexte conversationnel
    conversation_context, has_contextual_ref = get_conversation_context(question, current_session_id)
    memory_used = conversation_context is not None
    print(f"🧠 Mémoire: {'Utilisée' if memory_used else 'Non utilisée'} | Référence contextuelle: {has_contextual_ref}")
    
    # ÉTAPE 3: Traitement selon le type de question
    if question_type == "general":
        answer = call_mistral_without_rag_with_memory(question, conversation_context)
        cosine_score = 0.0
        grounded = False
        used_rag = False
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
    
    # Sauvegarde dans la conversation actuelle
    save_to_conversation(current_session_id, question, answer, cosine_score, grounded, question_type, used_rag, memory_used, short_summary, bullet_points)
    
    # Récupérer les données de conversation pour le titre
    _, conversation_data = load_conversation(current_session_id)
    conversation_title = conversation_data.get("title", "Nouvelle conversation") if conversation_data else "Nouvelle conversation"
    
    # Préparer la réponse
    response_data = {
        "response": answer,
        "cosine_score": round(cosine_score, 3),
        "confidence": round(cosine_score, 3),  # Ajouté pour compatibilité frontend
        "grounded_in_docs": grounded,
        "confidence_level": "haute" if grounded else "faible",
        "question_type": question_type,
        "classification_confidence": round(classification_confidence, 2),
        "used_rag": used_rag,
        "memory_used": memory_used,
        "has_contextual_reference": has_contextual_ref,
        "warning_displayed": answer.startswith("⚠️"),
        "session_id": current_session_id,
        "title": conversation_title
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

@app.route("/conversations", methods=["GET"])
def get_conversations():
    """Retourne la liste de toutes les conversations."""
    conversations = get_all_conversations()
    return jsonify({
        "conversations": conversations,
        "current_session": current_session_id
    })

@app.route("/conversation/<session_id>", methods=["GET"])
def get_conversation(session_id):
    """Retourne une conversation spécifique."""
    messages, conversation_data = load_conversation(session_id)
    
    if conversation_data:
        return jsonify({
            "session_id": session_id,
            "title": conversation_data.get("title", "Conversation"),
            "created_at": conversation_data.get("created_at"),
            "updated_at": conversation_data.get("updated_at"),
            "messages": messages
        })
    else:
        return jsonify({"error": "Conversation non trouvée"}), 404

@app.route("/switch-conversation", methods=["POST"])
def switch_conversation():
    """Change la session actuelle vers une autre conversation."""
    global current_session_id
    
    data = request.get_json() or {}
    new_session_id = data.get("session_id")
    
    if not new_session_id:
        return jsonify({"error": "session_id manquant"}), 400
    
    # Vérifier si la conversation existe
    conversation_file = get_conversation_file_path(new_session_id)
    if not os.path.exists(conversation_file):
        return jsonify({"error": "Conversation non trouvée"}), 404
    
    # Changer la session courante
    current_session_id = new_session_id
    print(f"🔄 Session changée vers: {current_session_id[:8]}...")
    
    # Retourner la conversation chargée
    messages, conversation_data = load_conversation(current_session_id)
    
    return jsonify({
        "session_id": current_session_id,
        "title": conversation_data.get("title", "Conversation") if conversation_data else "Conversation",
        "messages": messages
    })

@app.route("/new-conversation", methods=["POST"])
def new_conversation():
    """Crée une nouvelle conversation et la définit comme active."""
    global current_session_id
    
    new_session_id = create_new_conversation()
    
    if new_session_id:
        current_session_id = new_session_id
        return jsonify({
            "session_id": current_session_id,
            "title": "Nouvelle conversation",
            "message": "Nouvelle conversation créée"
        })
    else:
        return jsonify({"error": "Impossible de créer une nouvelle conversation"}), 500

@app.route("/delete-conversation/<session_id>", methods=["DELETE"])
def delete_conversation(session_id):
    """Supprime une conversation."""
    global current_session_id
    
    conversation_file = get_conversation_file_path(session_id)
    
    try:
        if os.path.exists(conversation_file):
            # Créer une sauvegarde avant suppression
            backup_file = conversation_file + f".deleted.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(conversation_file, backup_file)
            
            # Si c'est la conversation courante, créer une nouvelle
            if current_session_id == session_id:
                current_session_id = create_new_conversation()
            
            return jsonify({
                "message": "Conversation supprimée avec succès",
                "current_session": current_session_id
            })
        else:
            return jsonify({"error": "Conversation non trouvée"}), 404
            
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la suppression: {e}"}), 500

# Compatibilité avec les anciennes routes
@app.route("/session-history", methods=["GET"])
def session_history():
    """Retourne l'historique de la session actuelle (compatibilité)."""
    messages, conversation_data = load_conversation(current_session_id)
    
    return jsonify({
        "session_id": current_session_id,
        "title": conversation_data.get("title", "Conversation") if conversation_data else "Conversation",
        "messages": messages
    })

@app.route("/new-session", methods=["POST"])  
def new_session():
    """Alias pour new_conversation (compatibilité)."""
    return new_conversation()

# --- Lancement de l'application ---
if __name__ == "__main__":
    # Créer le dossier data s'il n'existe pas
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Créer la première conversation si aucune n'existe
    conversations = get_all_conversations()
    if not conversations:
        current_session_id = create_new_conversation()
    
    app.run(debug=True, port=5001)