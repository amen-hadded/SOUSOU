# Système de Prédiction des Prix Immobiliers en Tunisie

Projet Python combinant web scraping et machine learning pour prédire les prix de location d'appartements en Tunisie.

## Structure du projet

- **data.py** : Script de web scraping pour collecter les données depuis Lilkre.tn
- **app.py** : Application de machine learning pour analyser et prédire les prix
- **annonces_appartements.csv** : Base de données générée automatiquement
- **requirements.txt** : Dépendances Python

## Installation

1. Installer les dépendances :

```bash
pip install -r requirements.txt
```

2. Exécuter dans l'ordre :

```bash
# 1. Collecter les données
python data.py

# 2. Analyser et prédire
python app.py
```

## Fichier requirements.txt

```
requests==2.31.0
beautifulsoup4==4.12.2
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
```

## data.py - Collecte de données

### Fonctionnalités

- Scraping des annonces depuis Lilkre.tn
- Extraction des informations : prix, pièces, salles de bain, surface, gouvernorat
- Nettoyage automatique des données
- Export CSV

### Gouvernorats couverts

Ariana, Ben Arous, Bizerte, La Manouba, Monastir, Nabeul, Sfax, Sousse, Tunis

### Utilisation

```bash
python data.py
```

Génère le fichier `annonces_appartements.csv`

## app.py - Prédiction des prix

### Fonctionnalités

- Chargement et nettoyage des données
- Création de nouvelles variables (prix/m², surface/pièce)
- Entraînement d'un modèle de Régression Linéaire
- Évaluation des performances (R², MAE, RMSE)
- Prédictions personnalisées

### Modèle de Machine Learning

- **Algorithme** : Régression Linéaire
- **Features** : Pièces, Bains, Surface, Gouvernorat, surface_par_piece
- **Évaluation** : 80% entraînement, 20% test

### Utilisation

```bash
python app.py
```

## Workflow complet

1. **Collecte** :

```bash
python data.py
```

Scrape les données → génère `annonces_appartements.csv`

2. **Analyse** :

```bash
python app.py
```

Nettoie les données → entraîne le modèle → fait des prédictions

## Exemples de prédictions

| Pièces | SDB | Surface | Gouvernorat | Prix Estimé |
|--------|-----|---------|-------------|-------------|
| 2 | 1 | 70m² | Ariana | ~600 TND |
| 3 | 1 | 90m² | Tunis | ~850 TND |
| 4 | 2 | 120m² | Sousse | ~1100 TND |

## Personnalisation

### Ajouter un gouvernorat

Dans `data.py` :

```python
GOUVERNORATS = {
    "https://lilkre.tn/...": "Nouveau_Gouvernorat",
    # ...
}
```

### Modifier les features

Dans `app.py` :

```python
features = ['Pièces', 'Surface', 'Gouvernorat_encoded']
```

## Dépannage

### Problème : Erreur de scraping

Solution : Vérifier les sélecteurs CSS dans `data.py`

### Problème : Performance faible du modèle

Solution : Collecter plus de données ou ajuster les paramètres

## Auteur

- [Amen Allah Hadded](https://github.com/amen-hadded)


---

**Note** : Ce projet est éducatif. Les prédictions sont des estimations.
# SOUSOU
