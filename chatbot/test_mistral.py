#!/usr/bin/env python3
"""
Script de test pour vérifier la connexion à l'API Mistral
"""

from dotenv import load_dotenv
from openai import OpenAI
import os

# Charger les variables d'environnement
load_dotenv()

def test_mistral_connection():
    """Test de connexion à l'API Mistral"""
    
    # Récupération de la clé API
    api_key = os.getenv("MISTRAL_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Aucune clé API trouvée dans les variables d'environnement")
        return False
    
    print(f"🔑 Clé API trouvée: {api_key[:6]}***{api_key[-4:]}")
    
    # Configuration du client
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1"
        )
        print("✅ Client OpenAI configuré pour Mistral")
    except Exception as e:
        print(f"❌ Erreur de configuration du client: {e}")
        return False
    
    # Test de liste des modèles
    try:
        print("🔍 Test de connexion - Liste des modèles...")
        models = client.models.list()
        
        if hasattr(models, 'data') and len(models.data) > 0:
            print(f"✅ Connexion réussie! {len(models.data)} modèles disponibles:")
            for i, model in enumerate(models.data[:10]):  # Afficher les 10 premiers
                print(f"  {i+1}. {model.id}")
                if hasattr(model, 'created'):
                    print(f"     Créé: {model.created}")
        else:
            print("⚠️ Connexion OK mais aucun modèle trouvé")
            
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des modèles: {e}")
        print(f"Type d'erreur: {type(e).__name__}")
        return False
    
    # Test d'un appel de chat
    try:
        print("\n🤖 Test d'appel de chat...")
        response = client.chat.completions.create(
            model="mistral-small-2409",  # Ou essayez "mistral-medium-2505"
            messages=[
                {"role": "user", "content": "Bonjour, réponds juste 'Test réussi!'"}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        if response.choices and len(response.choices) > 0:
            answer = response.choices[0].message.content.strip()
            print(f"✅ Réponse reçue: {answer}")
            return True
        else:
            print("⚠️ Réponse vide reçue")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors de l'appel de chat: {e}")
        print(f"Type d'erreur: {type(e).__name__}")
        
        # Suggérer des modèles alternatifs
        try:
            print("\n🔄 Tentative avec un autre modèle...")
            response = client.chat.completions.create(
                model="mistral-medium-2505",
                messages=[
                    {"role": "user", "content": "Test"}
                ],
                max_tokens=10
            )
            print("✅ Modèle 'mistral-medium-2505' fonctionne!")
            return True
        except Exception as e2:
            print(f"❌ Échec avec modèle alternatif: {e2}")
            return False

if __name__ == "__main__":
    print("=== Test de connexion à l'API Mistral ===\n")
    success = test_mistral_connection()
    
    if success:
        print("\n🎉 Tous les tests sont passés! Votre configuration fonctionne.")
    else:
        print("\n💡 Suggestions de dépannage:")
        print("1. Vérifiez que votre clé API Mistral est correcte")
        print("2. Vérifiez votre fichier .env")
        print("3. Essayez de régénérer votre clé API sur https://console.mistral.ai/")
        print("4. Vérifiez votre connexion internet")