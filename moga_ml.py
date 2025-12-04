import random
import time
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1. Configuration du problème
# Nous voulons :
#   - Maximiser la Précision (Accuracy) -> Poids : 1.0
#   - Minimiser le Temps d'entraînement -> Poids : -1.0
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Chargement des données
data = load_digits()
X, y = data.data, data.target
# Split train/test pour validation finale si besoin, mais on utilisera CV dans l'évaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

toolbox = base.Toolbox()

# 2. Définition des Gènes (Hyperparamètres)
# Gene 0: n_estimators (10 à 200)
toolbox.register("attr_n_estimators", random.randint, 10, 200)
# Gene 1: max_depth (2 à 30)
toolbox.register("attr_max_depth", random.randint, 2, 30)
# Gene 2: min_samples_split (2 à 20)
toolbox.register("attr_min_samples_split", random.randint, 2, 20)
# Gene 3: max_features (0.1 à 1.0) - on utilisera random.uniform et on clippera
toolbox.register("attr_max_features", random.uniform, 0.1, 1.0)

# Structure de l'individu : [n_estimators, max_depth, min_samples_split, max_features]
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_n_estimators, toolbox.attr_max_depth, 
                  toolbox.attr_min_samples_split, toolbox.attr_max_features), n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 3. Fonction d'évaluation
def evaluate(individual):
    # Décodage des paramètres
    n_est = int(individual[0])
    max_d = int(individual[1])
    min_ss = int(individual[2])
    max_f = individual[3]
    
    # Contraintes de sécurité pour éviter les erreurs
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
        n_jobs=-1, # Utiliser tous les coeurs pour accélérer
        random_state=42
    )

    # Mesure du temps d'entraînement
    start_time = time.time()
    # On fait un cross-validation 3-fold pour avoir une accuracy robuste
    # Note: cross_val_score entraîne le modèle 3 fois.
    # Pour avoir le temps d'un seul entraînement représentatif, on peut diviser par 3 ou juste mesurer le tout.
    # Ici on mesure le temps total de la validation croisée comme proxy du coût de calcul.
    scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy')
    end_time = time.time()
    
    training_time = end_time - start_time
    accuracy = scores.mean()
    
    return accuracy, training_time

toolbox.register("evaluate", evaluate)

# 4. Opérateurs Génétiques
# Croisement (Crossover)
toolbox.register("mate", tools.cxTwoPoint)

# Mutation
# On doit définir une mutation personnalisée ou utiliser mutGaussian pour les floats, 
# mais ici on a un mix int/float. On va faire une mutation simple qui ré-initialise un gène.
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

toolbox.register("mutate", mutate_mixed, indpb=0.2)

# Sélection : NSGA-II est le standard pour le multi-objectif
toolbox.register("select", tools.selNSGA2)

def main():
    random.seed(42)
    
    # Paramètres de l'algo génétique
    NGEN = 5       # Nombre de générations (Augmenter à 20+ pour de meilleurs résultats)
    MU = 20        # Taille de la population (Augmenter à 50+ pour de meilleurs résultats)
    CXPB = 0.9     # Probabilité de croisement
    MUTPB = 0.1    # Probabilité de mutation
    
    # Création de la population initiale
    pop = toolbox.population(n=MU)
    
    # Statistiques pour le log
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    print("Début de l'évolution...")
    
    # Algorithme eaMuPlusLambda ou eaSimple. NSGA-II utilise souvent eaMuPlusLambda
    # Mais DEAP a une implémentation simple pour NSGA-II via select, on peut utiliser algorithms.eaMuPlusLambda
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=MU, 
                                             cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, 
                                             stats=stats, verbose=True)
                                             
    print("Fin de l'évolution.")
    
    # Récupération du Front de Pareto (les meilleures solutions non dominées)
    pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    
    print(f"\nNombre de solutions dans le front de Pareto : {len(pareto_front)}")
    
    # Affichage des résultats
    accuracies = [ind.fitness.values[0] for ind in pareto_front]
    times = [ind.fitness.values[1] for ind in pareto_front]
    
    # Tri pour l'affichage propre
    sorted_indices = np.argsort(accuracies)
    accuracies = np.array(accuracies)[sorted_indices]
    times = np.array(times)[sorted_indices]
    
    print("\nSolutions du Front de Pareto (Accuracy vs Time):")
    for acc, t, ind in zip(accuracies, times, np.array(pareto_front)[sorted_indices]):
        print(f"Acc: {acc:.4f}, Time: {t:.4f}s | Params: n_est={ind[0]}, depth={ind[1]}, split={ind[2]}, feat={ind[3]:.2f}")

    # Visualisation
    plt.figure(figsize=(10, 6))
    
    # Tous les individus finaux
    all_acc = [ind.fitness.values[0] for ind in pop]
    all_time = [ind.fitness.values[1] for ind in pop]
    plt.scatter(all_time, all_acc, c='gray', alpha=0.5, label='Population Finale')
    
    # Front de Pareto
    plt.scatter(times, accuracies, c='red', s=50, label='Front de Pareto')
    plt.plot(times, accuracies, c='red', linestyle='--', alpha=0.7)
    
    plt.xlabel("Temps d'entraînement (secondes) [Minimiser]")
    plt.ylabel("Précision (Accuracy) [Maximiser]")
    plt.title("Optimisation Multi-Objectif : Précision vs Temps (NSGA-II)")
    plt.legend()
    plt.grid(True)
    plt.savefig('pareto_front.png')
    print("\nGraphique sauvegardé sous 'pareto_front.png'")
    plt.show()

if __name__ == "__main__":
    main()
