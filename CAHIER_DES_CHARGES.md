# Cahier des Charges : Optimisation Multi-Objectif en Machine Learning

## 1. Présentation du Projet
**Titre** : Optimisation des Hyperparamètres d'un Modèle de Machine Learning par Algorithme Génétique Multi-Objectif (MOGA).  
**Cadre** : Module d'Algorithmique Avancée.  
**Type** : Projet de développement et d'analyse algorithmique.

## 2. Contexte et Problématique
Dans le domaine du Machine Learning, la performance d'un modèle dépend fortement de ses hyperparamètres. Cependant, l'amélioration de la précision (Accuracy) se fait souvent au détriment de la complexité du modèle et donc du temps d'entraînement ou d'inférence.
Il s'agit d'un problème d'optimisation multi-objectif où l'on cherche à minimiser le temps de calcul tout en maximisant la précision.

**Problématique** : Comment identifier automatiquement les meilleures configurations d'un modèle (Front de Pareto) sans tester exhaustivement toutes les combinaisons ?

## 3. Objectifs du Projet
L'objectif principal est de développer une application Python capable de :
1.  **Explorer** l'espace des hyperparamètres d'un classifieur (Random Forest).
2.  **Optimiser** simultanément deux objectifs conflictuels :
    *   Maximiser la Précision (Accuracy).
    *   Minimiser le Temps d'Entraînement (Training Time).
3.  **Fournir** un ensemble de solutions optimales (Front de Pareto) à l'utilisateur.

## 4. Spécifications Fonctionnelles

### 4.1. Module de Données
*   Le système doit charger un jeu de données standard (ex: *Digits* de Scikit-learn ou *MNIST*).
*   Le jeu de données doit être séparé en ensembles d'entraînement et de test (ou utiliser la validation croisée).

### 4.2. Modèle de Machine Learning
*   Algorithme cible : **Random Forest Classifier**.
*   Hyperparamètres à optimiser (Gènes) :
    *   `n_estimators` (Nombre d'arbres) : Entier [10, 200].
    *   `max_depth` (Profondeur maximale) : Entier [2, 30].
    *   `min_samples_split` : Entier [2, 20].
    *   `max_features` : Flottant [0.1, 1.0].

### 4.3. Algorithme d'Optimisation (MOGA)
*   Utilisation de l'algorithme **NSGA-II** (Non-dominated Sorting Genetic Algorithm II).
*   **Encodage** : Représentation mixte (entiers et flottants) des individus.
*   **Fonction de Fitness** :
    *   Objectif 1 : Accuracy (à maximiser).
    *   Objectif 2 : Temps d'exécution (à minimiser).
*   **Opérateurs** :
    *   Sélection : Tournoi ou NSGA-II selection.
    *   Croisement (Crossover) : À deux points ou uniforme.
    *   Mutation : Modification aléatoire d'un gène.

### 4.4. Visualisation et Sortie
*   Affichage console de l'évolution (génération par génération).
*   Génération d'un graphique **Scatter Plot** :
    *   Axe X : Temps d'entraînement.
    *   Axe Y : Précision.
    *   Mise en évidence du **Front de Pareto** (solutions non dominées).
*   Sauvegarde des résultats (graphique).

## 5. Spécifications Techniques

### 5.1. Environnement de Développement
*   **Langage** : Python 3.x.
*   **Système d'exploitation** : Windows / Linux / MacOS.

### 5.2. Bibliothèques Principales
*   **Scikit-learn** : Pour le modèle Random Forest, le chargement des données et l'évaluation.
*   **DEAP** (Distributed Evolutionary Algorithms in Python) : Pour l'implémentation de NSGA-II.
*   **Matplotlib** : Pour la visualisation des données.
*   **NumPy** : Pour les calculs matriciels.

## 6. Livrables Attendus
1.  **Code Source** : Scripts Python commentés (`moga_ml.py`).
2.  **Fichier de dépendances** : `requirements.txt`.
3.  **Documentation** : `README.md` expliquant l'installation et l'exécution.
4.  **Rapport (optionnel)** : Analyse des résultats et du front de Pareto obtenu.

## 7. Critères de Réussite
*   Le script s'exécute sans erreur.
*   L'algorithme converge vers des solutions optimisées.
*   Le graphique montre clairement le compromis entre temps et précision (forme convexe typique du front de Pareto).
