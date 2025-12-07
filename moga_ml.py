import random
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from sklearn.datasets import load_digits, load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 1. Configuration du problème
# Nous voulons :
#   - Maximiser la Précision (Accuracy) -> Poids : 1.0
#   - Minimiser le Temps d'entraînement -> Poids : -1.0
# Check if creator has been initialized to avoid errors on reload
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti)

def get_dataset(name):
    if name == 'digits':
        data = load_digits()
    elif name == 'iris':
        data = load_iris()
    elif name == 'wine':
        data = load_wine()
    elif name == 'breast_cancer':
        data = load_breast_cancer()
    else:
        raise ValueError(f"Dataset inconnu : {name}")
    return data.data, data.target

def setup_toolbox():
    toolbox = base.Toolbox()
    
    # 2. Définition des Gènes (Hyperparamètres)
    # Gene 0: n_estimators (10 à 200)
    toolbox.register("attr_n_estimators", random.randint, 10, 200)
    # Gene 1: max_depth (2 à 30)
    toolbox.register("attr_max_depth", random.randint, 2, 30)
    # Gene 2: min_samples_split (2 à 20)
    toolbox.register("attr_min_samples_split", random.randint, 2, 20)
    # Gene 3: max_features (0.1 à 1.0)
    toolbox.register("attr_max_features", random.uniform, 0.1, 1.0)

    # Structure de l'individu : [n_estimators, max_depth, min_samples_split, max_features]
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_n_estimators, toolbox.attr_max_depth, 
                      toolbox.attr_min_samples_split, toolbox.attr_max_features), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Opérateurs Génétiques
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate_mixed, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    
    return toolbox

# Mutation personnalisée
def mutate_mixed(individual, indpb):
    # Mutation pour n_estimators
    if random.random() < indpb:
        individual[0] = random.randint(10, 200)
    # Mutation pour max_depth
    if random.random() < indpb:
        individual[1] = random.randint(2, 30)
    # Mutation pour min_samples_split
    if random.random() < indpb:
        individual[2] = random.randint(2, 20)
    # Mutation pour max_features
    if random.random() < indpb:
        individual[3] = random.uniform(0.1, 1.0)
    return individual,

def evaluate(individual, X, y):
    # Décodage des paramètres
    n_est = int(individual[0])
    max_d = int(individual[1])
    min_ss = int(individual[2])
    max_f = individual[3]
    
    # Contraintes de sécurité
    if n_est < 1: n_est = 1
    if max_d < 1: max_d = 1
    if min_ss < 2: min_ss = 2
    if max_f <= 0: max_f = 0.1
    if max_f > 1.0: max_f = 1.0

    clf = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=max_d,
        min_samples_split=min_ss,
        max_features=max_f,
        n_jobs=-1, 
        random_state=42
    )

    start_time = time.time()
    # Cross-validation 3-fold
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    end_time = time.time()
    
    training_time = end_time - start_time
    accuracy = scores.mean()
    
    return accuracy, training_time

def run_optimization(dataset_name='digits', ngen=5, pop_size=20):
    """
    Exécute l'optimisation MOGA et retourne les résultats.
    """
    random.seed(42)
    
    # Chargement des données
    X, y = get_dataset(dataset_name)
    
    # Configuration de la toolbox
    toolbox = setup_toolbox()
    toolbox.register("evaluate", evaluate, X=X, y=y)
    
    # Paramètres
    CXPB = 0.9
    MUTPB = 0.1
    
    # Population initiale
    pop = toolbox.population(n=pop_size)
    
    # Statistiques
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    # Algorithme
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size, 
                                             cxpb=CXPB, mutpb=MUTPB, ngen=ngen, 
                                             stats=stats, verbose=True)
    
    # Front de Pareto
    pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    
    # Extraction des résultats (Pareto)
    pareto_results = []
    for ind in pareto_front:
        pareto_results.append({
            "n_estimators": ind[0],
            "max_depth": ind[1],
            "min_samples_split": ind[2],
            "max_features": ind[3],
            "accuracy": ind.fitness.values[0],
            "training_time": ind.fitness.values[1]
        })
    
    # Extraction de toute la population (pour le scatter plot complet)
    population_results = []
    for ind in pop:
        population_results.append({
            "accuracy": ind.fitness.values[0],
            "training_time": ind.fitness.values[1]
        })
        
    return pareto_results, population_results

def main():
    parser = argparse.ArgumentParser(description="Optimisation MOGA pour Random Forest")
    parser.add_argument("--dataset", type=str, default="digits", 
                        choices=["digits", "iris", "wine", "breast_cancer"],
                        help="Jeu de données à utiliser (default: digits)")
    parser.add_argument("--ngen", type=int, default=5, help="Nombre de générations (default: 5)")
    parser.add_argument("--pop-size", type=int, default=20, help="Taille de la population (default: 20)")
    parser.add_argument("--output-plot", type=str, default="pareto_front.png", help="Fichier de sortie pour le graphique")
    parser.add_argument("--output-csv", type=str, default="pareto_results.csv", help="Fichier de sortie pour les résultats CSV")
    
    args = parser.parse_args()

    print(f"Chargement du dataset : {args.dataset}")
    print(f"Début de l'évolution (Pop: {args.pop_size}, Gen: {args.ngen})...")
    
    pareto_results, population_results = run_optimization(args.dataset, args.ngen, args.pop_size)
    
    print("Fin de l'évolution.")
    print(f"\nNombre de solutions dans le front de Pareto : {len(pareto_results)}")
    
    # Tri par accuracy
    pareto_results.sort(key=lambda x: x["accuracy"])
    
    print("\nSolutions du Front de Pareto (Accuracy vs Time):")
    for res in pareto_results:
        print(f"Acc: {res['accuracy']:.4f}, Time: {res['training_time']:.4f}s | "
              f"Params: n_est={res['n_estimators']}, depth={res['max_depth']}, "
              f"split={res['min_samples_split']}, feat={res['max_features']:.2f}")

    # Sauvegarde CSV
    df_results = pd.DataFrame(pareto_results)
    df_results.to_csv(args.output_csv, index=False)
    print(f"\nRésultats sauvegardés dans '{args.output_csv}'")

    # Visualisation
    plt.figure(figsize=(10, 6))
    
    # Population finale
    all_acc = [res['accuracy'] for res in population_results]
    all_time = [res['training_time'] for res in population_results]
    plt.scatter(all_time, all_acc, c='gray', alpha=0.5, label='Population Finale')
    
    # Front de Pareto
    pareto_acc = [res['accuracy'] for res in pareto_results]
    pareto_time = [res['training_time'] for res in pareto_results]
    
    plt.scatter(pareto_time, pareto_acc, c='red', s=50, label='Front de Pareto')
    plt.plot(pareto_time, pareto_acc, c='red', linestyle='--', alpha=0.7)
    
    plt.xlabel("Temps d'entraînement (secondes) [Minimiser]")
    plt.ylabel("Précision (Accuracy) [Maximiser]")
    plt.title(f"Optimisation Multi-Objectif ({args.dataset}) : Précision vs Temps")
    plt.legend()
    plt.grid(True)
    plt.savefig(args.output_plot)
    print(f"Graphique sauvegardé sous '{args.output_plot}'")

if __name__ == "__main__":
    main()
