import numpy as np
import matplotlib.pyplot as plt
import random

# Given Data 
Q = np.array([5, 7, 8])  
W = np.array([0.2, 0.35, 0.15])  

P = np.array([
    [0.8, 0.7, 0.0],
    [0.7, 0.0, 0.8],
    [0.0, 0.8, 0.7]
])

U = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

# given GA Parameters 
POPULATION_SIZE = 20
MAX_ITERATIONS = 60
MUTATION_RATE = 0.1

def calc_fitness(chromosome):
    X = np.array(chromosome).reshape(3, 3)
    for i in range(3):
        if np.sum(U[i] * X[i]) > Q[i]:
            return 0.0  # Invalid 
        
    raw_damage = np.sum(W * P * X)
    max_damage = np.sum(
     W.reshape(3,1) *
     P *
     np.array(Q).reshape(3,1)
     )
    fitness_value = raw_damage / max_damage 
    return fitness_value

def initialize_population():
    pop = []
    while len(pop) < POPULATION_SIZE:
        chrom = []
        for i in range(3):
            alloc = [0, 0, 0]
            total_weapons = random.randint(0, Q[i])
            for _ in range(total_weapons):
                target = random.choice([0, 1, 2])
                if U[i][target] == 1:  
                    alloc[target] += 1
            chrom.extend(alloc)
        pop.append(chrom)
    return pop

def crossover(parent1,parent2):
    p1 = np.array(parent1).reshape(3,3)
    p2 = np.array(parent2).reshape(3,3)
    child1 = p1.copy()
    child2 = p2.copy()
    cols = random.sample([0,1,2], random.randint(1,2))
    for c in cols:
        child1[:,c] = p2[:,c]
        child2[:,c] = p1[:,c]
    return child1.flatten().tolist(), child2.flatten().tolist()

def mutate(chromosome):
    chromosome = chromosome.copy()
    for _ in range(random.randint(1, 3)):
        if random.random() < MUTATION_RATE:
            row = random.randint(0, 2)
            col = random.randint(0, 2)
            if U[row][col] == 1:
                row_start = row * 3
                row_vals = chromosome[row_start:row_start+3]
                remaining = Q[row] - sum(row_vals) + row_vals[col]
                chromosome[row_start + col] = random.randint(0, remaining)
    return chromosome

# EXECUTE GA
population = initialize_population()
best_fitness_history = []
best_global_chromosome = None
best_global_fitness = 0.0

for generation in range(MAX_ITERATIONS):
    fitness_scores = [calc_fitness(ind) for ind in population]
    
    # Finding Best of Current Generation
    gen_best_fitness = max(fitness_scores)
    best_idx = fitness_scores.index(gen_best_fitness)
    current_best_chromosome = population[best_idx]
    
    best_fitness_history.append(gen_best_fitness)
    
    if gen_best_fitness > best_global_fitness:
        best_global_fitness = gen_best_fitness
        best_global_chromosome = current_best_chromosome
        
    # Output of every iteration
    num_weapons = 3
    num_targets = 3
    X_best = np.array(current_best_chromosome).reshape(3, 3)
    damage_per_target = np.sum(P * X_best, axis=0) 
    overall_damage = sum(damage_per_target) / num_targets
    max_damage = 0
    for i in range(3):
     max_damage += Q[i] * np.max(P[i])
    normalized_damage = overall_damage / max_damage

    assignment_matrix = [
    current_best_chromosome[i * num_targets : (i + 1) * num_targets] 
    for i in range(num_weapons)
    ]
    print(f"Iteration {generation + 1}:")
    print("Weapon Assignment Matrix:")
    for row in assignment_matrix:
     print(row)
    print(f"Overall Damage: {normalized_damage:.2f}")
    print(f"Fitness Score : {gen_best_fitness:.4f}")
    print("-" * 75)

    # Selection 
    selected = random.choices(population, weights=[f + 0.01 for f in fitness_scores], k=POPULATION_SIZE)
    
    # Crossover and Mutation
    next_generation = [current_best_chromosome.copy()]
    while len(next_generation) < POPULATION_SIZE:
     p1, p2 = random.sample(selected, 2)
     c1, c2 = crossover(p1, p2)
     c1 = mutate(c1)
     c2 = mutate(c2)
     next_generation.append(c1)
     if len(next_generation) < POPULATION_SIZE:
        next_generation.append(c2)
    population = next_generation

optimal_X = np.array(best_global_chromosome).reshape(3, 3)

# VISUALIZATION
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Stage 1: GA Convergence
ax1.plot(range(MAX_ITERATIONS), best_fitness_history, color='b', linewidth=2)
ax1.set_title("Stage 1: GA Convergence")
ax1.set_xlabel("Generations")
ax1.set_ylabel("Fitness")
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Stage 2: Defensive Distribution
x = np.linspace(0, 50, 100)
y = np.linspace(0, 50, 100)
X_grid, Y_grid = np.meshgrid(x, y)

Z = np.zeros_like(X_grid)
for i in range(3):
    for j in range(3):
        if optimal_X[i][j] > 0:
            cx, cy = random.uniform(10, 40), random.uniform(10, 40)
            Z += optimal_X[i][j] * np.exp(-((X_grid - cx)**2 + (Y_grid - cy)**2) / 20)

contour = ax2.contour(X_grid, Y_grid, Z, levels=10, cmap='jet', linewidths=1.5)
ax2.set_title("Stage 2: Defensive Positions (Contour)")
ax2.set_xlabel("X Coordinates")
ax2.set_ylabel("Y Coordinates")

plt.tight_layout()
plt.show()

# comparison
print("\nRESULTS COMPARISON")
print("Paper's Stated Optimal X:")
print(np.array([[1, 2, 2], [4, 0, 2], [0, 3, 1]]))
print("Paper's Stated Fitness: 0.6456\n")
print("Algorithm Computed Optimal X:")
print(optimal_X)
print(f"Algorithm Computed Fitness: {best_global_fitness:.4f}")