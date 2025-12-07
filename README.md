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
2. Lancer le script avec les options par défaut :
   ```bash
   python moga_ml.py
   ```

## Options Avancées
Le script supporte plusieurs arguments en ligne de commande pour personnaliser l'exécution :

| Argument | Description | Valeur par défaut | Choix possibles |
|----------|-------------|-------------------|-----------------|
| `--dataset` | Jeu de données à utiliser | `digits` | `digits`, `iris`, `wine`, `breast_cancer` |
| `--ngen` | Nombre de générations | `5` | Entier > 0 |
| `--pop-size` | Taille de la population | `20` | Entier > 0 |
| `--output-plot` | Nom du fichier graphique de sortie | `pareto_front.png` | Chemin de fichier |
| `--output-csv` | Nom du fichier CSV de résultats | `pareto_results.csv` | Chemin de fichier |

### Exemples d'utilisation

**Utiliser le dataset Iris avec 50 générations :**
```bash
python moga_ml.py --dataset iris --ngen 50
```

**Augmenter la taille de la population pour le dataset Breast Cancer :**
```bash
python moga_ml.py --dataset breast_cancer --pop-size 100 --ngen 20
```

**Sauvegarder les résultats dans des fichiers spécifiques :**
```bash
python moga_ml.py --output-plot mon_graphique.png --output-csv mes_resultats.csv
```

## Détails Techniques
Nous utilisons la librairie `DEAP` pour l'algorithme génétique et `scikit-learn` pour le modèle de Machine Learning (Random Forest). L'algorithme cherche la meilleure combinaison d'hyperparamètres (nombre d'arbres, profondeur max, etc.).
