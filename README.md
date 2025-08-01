# Seizure Detection and Analysis Project

## Description
Ce projet est un système d'analyse et de détection de crises d'épilepsie utilisant l'intelligence artificielle. Il comprend plusieurs composants pour l'augmentation de données, la prédiction, l'analyse et une interface de chatbot pour l'interaction avec les utilisateurs.

## Structure du Projet

```
seizure/
├── README.md                    # Documentation du projet
├── requirements.txt             # Dépendances Python
├── chat_history.json           # Historique des conversations
├── chatbot/                    # Interface chatbot
│   ├── app.py                  # Application Flask principale
│   ├── test_mistral.py         # Tests pour le modèle Mistral
│   ├── data/                   # Données du chatbot
│   ├── static/                 # Fichiers statiques (CSS, JS)
│   └── templates/              # Templates HTML
├── data_augmentation/          # Modules d'augmentation de données
├── framework/                  # Framework principal
├── prediction/                 # Modules de prédiction
└── review/                     # Modules de révision et analyse
```

## Fonctionnalités

- **Détection de crises** : Analyse des signaux EEG pour détecter les crises d'épilepsie
- **Augmentation de données** : Techniques pour enrichir les datasets d'entraînement
- **Prédiction** : Modèles d'IA pour prédire les crises
- **Interface chatbot** : Interface conversationnelle pour interagir avec le système
- **Analyse et révision** : Outils pour l'analyse des résultats

## Installation

1. Clonez le repository :
```bash
git clone https://github.com/Safaya-co/seizure.git
cd seizure
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

### Lancement du Chatbot
Pour démarrer l'interface chatbot :
```bash
cd chatbot
python app.py
```

L'application sera accessible à l'adresse `http://localhost:5000`

### Tests
Pour exécuter les tests :
```bash
cd chatbot
python test_mistral.py
```

## Technologies Utilisées

- **Python** : Langage principal
- **Flask** : Framework web pour le chatbot
- **Mistral AI** : Modèle de langage pour le chatbot
- **PyTorch** : Framework de deep learning
- **Embeddings BGE** : Modèles d'embeddings pour la recherche sémantique

## Configuration

Le projet utilise des embeddings pré-entraînés et des données de récupération stockées dans le dossier `chatbot/data/`. Assurez-vous que les fichiers suivants sont présents :
- `embeddings_bge_v15.pt`
- `retrieval_data_bge_v15.pkl`

## Contribution

Pour contribuer au projet :
1. Forkez le repository
2. Créez une branche pour votre fonctionnalité
3. Committez vos changements
4. Poussez vers la branche
5. Ouvrez une Pull Request

## Licence

Ce projet est développé par Safaya-co pour l'analyse et la détection de crises d'épilepsie.

## Contact
by github

## document de recherche:
https://docs.google.com/document/d/1GhA1Yfc1vRqM34yANVCLZH-TiwhxbQcjhHNFWS8oyZg/edit?usp=sharing
