# Projet MOGA : Optimisation Multi-Objectif en Machine Learning

Ce projet démontre l'utilisation d'un Algorithme Génétique Multi-Objectif (MOGA), spécifiquement NSGA-II, pour optimiser deux objectifs conflictuels dans un modèle de Machine Learning :
1. **Maximiser la Précision (Accuracy)**
2. **Minimiser le Temps d'Entraînement**

## Structure
- `moga_ml.py` : Le script principal contenant l'implémentation.
- `requirements.txt` : Les dépendances nécessaires.

## Comment exécuter
1. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
2. Lancer le script :
   ```bash
   python moga_ml.py
   ```

## Détails Techniques
Nous utilisons la librairie `DEAP` pour l'algorithme génétique et `scikit-learn` pour le modèle de Machine Learning (Random Forest). L'algorithme cherche la meilleure combinaison d'hyperparamètres (nombre d'arbres, profondeur max, etc.).
