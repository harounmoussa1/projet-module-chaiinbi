import random
import time
import numpy as np
from deap import base, creator, tools
from sklearn.datasets import load_digits, load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))

if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti)

def get_dataset(name):
    datasets = {
        'digits': load_digits,
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer
    }
    if name not in datasets:
        raise ValueError(f"Dataset inconnu : {name}")
    data = datasets[name]()
    return data.data, data.target

def setup_toolbox():
    toolbox = base.Toolbox()
    toolbox.register("attr_n_estimators", random.randint, 10, 200)
    toolbox.register("attr_max_depth", random.randint, 2, 30)
    toolbox.register("attr_min_samples_split", random.randint, 2, 20)
    toolbox.register("attr_max_features", random.uniform, 0.1, 1.0)
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (
            toolbox.attr_n_estimators,
            toolbox.attr_max_depth,
            toolbox.attr_min_samples_split,
            toolbox.attr_max_features
        ),
        n=1
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate_mixed, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    return toolbox

def mutate_mixed(individual, indpb):
    if random.random() < indpb:
        individual[0] = random.randint(10, 200)
    if random.random() < indpb:
        individual[1] = random.randint(2, 30)
    if random.random() < indpb:
        individual[2] = random.randint(2, 20)
    if random.random() < indpb:
        individual[3] = random.uniform(0.1, 1.0)
    return individual,

def evaluate(individual, X, y):
    n_est = int(individual[0])
    max_d = int(individual[1])
    min_ss = int(individual[2])
    max_f = max(0.1, min(1.0, individual[3]))
    clf = RandomForestClassifier(
        n_estimators=max(1, n_est),
        max_depth=max(1, max_d),
        min_samples_split=max(2, min_ss),
        max_features=max_f,
        n_jobs=-1,
        random_state=42
    )
    start_time = time.time()
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    training_time = time.time() - start_time
    return scores.mean(), training_time

def run_optimization(dataset_name='digits', ngen=5, pop_size=20, callback=None):
    random.seed(42)
    if callback:
        callback("phase", {"phase": "initialization"})
    X, y = get_dataset(dataset_name)
    toolbox = setup_toolbox()
    if callback:
        callback("phase", {"phase": "sampling"})
    generation_points = []
    current_generation = {"gen": 0}
    def evaluate_with_callback(individual):
        acc, t = evaluate(individual, X, y)
        gen = current_generation["gen"]
        if gen >= len(generation_points):
            generation_points.append([])
        generation_points[gen].append({
            "accuracy": float(acc),
            "training_time": float(t),
            "individual": [
                int(individual[0]),
                int(individual[1]),
                int(individual[2]),
                float(individual[3])
            ]
        })
        return acc, t
    toolbox.register("evaluate", evaluate_with_callback)
    if callback:
        callback("phase", {"phase": "evaluation"})
    pop = toolbox.population(n=pop_size)
    if callback:
        callback("generation_start", {"generation": 0, "total": ngen})
    current_generation["gen"] = 0
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    if callback:
        for pt in generation_points[0]:
            callback("point", {**pt, "generation": 0})
    if callback:
        callback("generation_end", {"generation": 0, "total": ngen})
    if callback:
        callback("phase", {"phase": "model_update"})
    for gen in range(1, ngen + 1):
        current_generation["gen"] = gen
        if callback:
            callback("generation_start", {"generation": gen, "total": ngen})
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.9:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values
        for m in offspring:
            if random.random() < 0.1:
                toolbox.mutate(m)
                del m.fitness.values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = toolbox.select(pop + offspring, pop_size)
        if callback and gen < len(generation_points):
            for pt in generation_points[gen]:
                callback("point", {**pt, "generation": gen})
        if callback:
            callback("generation_end", {"generation": gen, "total": ngen})
    if callback:
        callback("phase", {"phase": "pareto"})
    pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    pareto_results = []
    for ind in pareto_front:
        pareto_results.append({
            "n_estimators": int(ind[0]),
            "max_depth": int(ind[1]),
            "min_samples_split": int(ind[2]),
            "max_features": float(ind[3]),
            "accuracy": float(ind.fitness.values[0]),
            "training_time": float(ind.fitness.values[1])
        })
    non_pareto = []
    for ind in pop:
        if ind not in pareto_front:
            non_pareto.append({
                "accuracy": float(ind.fitness.values[0]),
                "training_time": float(ind.fitness.values[1])
            })
    population_results = [
        {
            "accuracy": float(ind.fitness.values[0]),
            "training_time": float(ind.fitness.values[1])
        }
        for ind in pop
    ]
    if callback:
        callback("results", {
            "pareto": pareto_results,
            "non_pareto": non_pareto,
            "population": population_results
        })
    return pareto_results, population_results
