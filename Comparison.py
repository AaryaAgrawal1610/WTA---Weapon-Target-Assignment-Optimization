import numpy as np
import matplotlib.pyplot as plt
import random
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.preprocessing import MinMaxScaler

# COMMON DATA
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

POPULATION_SIZE = 30
SWARM_SIZE = 20
MAX_ITERATIONS = 60
MUTATION_RATE = 0.08

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def calc_normalized_damage(X):
    X = np.array(X).reshape(3,3)
    weighted_damage = np.sum(W.reshape(3,1) * P * X)
    max_damage = np.sum(
        W.reshape(3,1) *
        P *
        np.array(Q).reshape(3,1)
    )
    normalized_damage = weighted_damage / max_damage
    return weighted_damage, normalized_damage

def calc_fitness(chromosome):
    X = np.array(chromosome).reshape(3, 3)
    for i in range(3):
        if np.sum(U[i] * X[i]) > Q[i]:
            return 0.0  
        
    raw_damage = np.sum(W * P * X)
    max_damage = np.sum(
     W.reshape(3,1) *
     P *
     np.array(Q).reshape(3,1)
     )
    fitness_value = raw_damage / max_damage 
    return fitness_value

def generate_random_solution():
    chrom = []
    for i in range(3):
        alloc = [0, 0, 0]
        total_weapons = random.randint(0, Q[i])
        for _ in range(total_weapons):
            target = random.randint(0, 2)
            if U[i][target] == 1:
                alloc[target] += 1
        chrom.extend(alloc)
    return np.array(chrom)

def crossover(parent1, parent2):
    point = random.randint(1, 8)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

def mutate(chromosome):
    chromosome = chromosome.copy()
    if random.random() < MUTATION_RATE:
        idx = random.randint(0, 8)
        row = idx // 3
        col = idx % 3
        if U[row][col] == 1:
            chromosome[idx] = random.randint(0, Q[row])
    return chromosome

class Particle:
    def __init__(self):
        self.position = generate_random_solution()
        self.velocity = np.random.uniform(
            -0.5,
            0.5,
            9
        )
        self.pbest_position = self.position.copy()
        self.pbest_fitness = calc_fitness(
            self.position
        )

def local_search(solution):
    best = solution.copy()
    best_fit = calc_fitness(best)
    for _ in range(100):
        candidate = best.copy()
        idx = random.randint(0, 8)
        row = idx // 3
        col = idx % 3
        if U[row][col] == 1:
            change = random.choice([-1, 1])
            candidate[idx] = max(
                0,
                candidate[idx] + change
            )
            X = candidate.reshape(3, 3)
            for r in range(3):
                total = np.sum(X[r])
                if total > Q[r]:
                    factor = Q[r] / total
                    X[r] = np.floor(X[r] * factor)
            candidate = X.flatten().astype(int)
            fit = calc_fitness(candidate)
            if fit > best_fit:
                best = candidate.copy()
                best_fit = fit
    return best

# GA 
def run_ga():
    POPULATION_SIZE_GA = 10
    population = [generate_random_solution()
                  for _ in range(POPULATION_SIZE_GA)]
    best_history = []
    dmg_history = []
    best_solution = None
    best_fitness = -1
    for iteration in range(MAX_ITERATIONS):
        fitness_scores = [calc_fitness(ind)
                          for ind in population]
        best_idx = np.argmax(fitness_scores)
        current_best = population[best_idx]
        current_fitness = fitness_scores[best_idx]
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_solution = current_best.copy()
        best_history.append(best_fitness)
        
        _, nd = calc_normalized_damage(best_solution)
        dmg_history.append(nd)

        selected = random.choices(
            population,
            weights=[f + 0.01 for f in fitness_scores],
            k=POPULATION_SIZE_GA
        )

        elite_count = 2
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        next_generation = [
            population[i].copy()
            for i in elite_indices
        ]
        while len(next_generation) < POPULATION_SIZE_GA:
            p1, p2 = random.sample(selected, 2)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            next_generation.append(c1)
            if len(next_generation) < POPULATION_SIZE_GA:
                next_generation.append(c2)
        population = next_generation

    return best_solution.reshape(3,3), best_fitness, best_history, dmg_history

# PSO 
def run_pso():
    W_INERTIA = 0.4
    C1 = 1.5
    C2 = 1.5 
    swarm = [Particle() for _ in range(SWARM_SIZE)]
    best_history = []
    dmg_history = []
    gbest_position = swarm[0].position.copy()
    gbest_fitness = calc_fitness(gbest_position)
    for p in swarm:
        fit = calc_fitness(p.position)
        if fit > gbest_fitness:
            gbest_fitness = fit
            gbest_position = p.position.copy()
            
    for iteration in range(MAX_ITERATIONS):
        for p in swarm:
            r1 = np.random.rand(9)
            r2 = np.random.rand(9)
            p.velocity = (
              W_INERTIA * p.velocity
              + C1 * r1 * (p.pbest_position - p.position)
              + C2 * r2 * (gbest_position - p.position)
            )
            velocity_limit = 2
            p.velocity = np.clip(
              p.velocity,
             -velocity_limit,
             velocity_limit
            )

            p.position = p.position + 0.8 * p.velocity
            p.position = np.round(p.position).astype(int)
            p.position = np.maximum(p.position, 0)

            for j in range(9):
                row = j // 3
                col = j % 3
                if U[row][col] == 0:
                    p.position[j] = 0
            
            X = p.position.reshape(3,3)
            for r in range(3):
                total = np.sum(X[r])
                if total > Q[r]:
                    factor = Q[r] / total
                    X[r] = np.floor(X[r] * factor)

            p.position = X.flatten().astype(int)
            fitness = calc_fitness(p.position)

            if fitness > p.pbest_fitness:
                p.pbest_fitness = fitness
                p.pbest_position = p.position.copy()

            if fitness > gbest_fitness:
                gbest_fitness = fitness
                gbest_position = p.position.copy()

        best_history.append(gbest_fitness)
        _, nd = calc_normalized_damage(gbest_position)
        dmg_history.append(nd)
        
    return gbest_position.reshape(3,3), gbest_fitness, best_history, dmg_history

# HYBRID GA-PSO
def run_gapso():
    W_INERTIA = 0.9
    C1 = 2.5
    C2 = 2.5
    population = [generate_random_solution()
                  for _ in range(POPULATION_SIZE)]
    GA_GENERATIONS = 25

    for _ in range(GA_GENERATIONS):
        fitness_scores = [calc_fitness(ind)
                          for ind in population]
        best_idx = np.argmax(fitness_scores)
        elite = population[best_idx].copy()
        selected = random.choices(
            population,
            weights=[f + 0.01 for f in fitness_scores],
            k=POPULATION_SIZE
        )
        next_generation = [elite.copy()]
        while len(next_generation) < POPULATION_SIZE:
            p1, p2 = random.sample(selected, 2)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            next_generation.append(c1)
            if len(next_generation) < POPULATION_SIZE:
                next_generation.append(c2)
        next_generation[0] = elite
        population = next_generation

    population = sorted(
        population,
        key=lambda x: calc_fitness(x),
        reverse=True
    )

    elite = population[:10]
    new_random = [
        generate_random_solution()
        for _ in range(10)
    ]

    population = elite + new_random

    while len(population) < SWARM_SIZE:
      population.append(
        population[random.randint(0,4)].copy()
      )

    swarm = []

    for solution in population:
        particle = Particle()
        particle.position = solution.copy()
        particle.pbest_fitness = calc_fitness(solution)
        particle.velocity = np.random.uniform(-2,2,9)
        particle.pbest_position = solution.copy()
        particle.pbest_fitness = calc_fitness(solution)
        swarm.append(particle)

    best_history = []
    dmg_history = []
    gbest_position = swarm[0].position.copy()
    gbest_fitness = calc_fitness(gbest_position)

    for iteration in range(MAX_ITERATIONS):
        for p in swarm:
            r1 = np.random.rand(9)
            r2 = np.random.rand(9)
            p.velocity = (
                W_INERTIA * p.velocity
                + C1 * r1 *
                (p.pbest_position - p.position)
                + C2 * r2 *
                (gbest_position - p.position)
            )

            p.position = p.position + p.velocity
            p.position = np.round(p.position).astype(int)
            p.position = np.maximum(p.position, 0)

            for j in range(9):
                row = j // 3
                col = j % 3
                if U[row][col] == 0:
                    p.position[j] = 0

            X = p.position.reshape(3,3)

            for r in range(3):
                total = np.sum(X[r])
                if total > Q[r]:
                    factor = Q[r] / total
                    X[r] = np.floor(X[r] * factor)

            p.position = X.flatten().astype(int)
            if random.random() < 0.05:
             idx = random.randint(0,8)
             row = idx//3
             p.position[idx] = random.randint(
                0,
                Q[row]
              )
            p.position = local_search(p.position)
            fitness = calc_fitness(p.position)

            if fitness > p.pbest_fitness:
                p.pbest_fitness = fitness
                p.pbest_position = p.position.copy()

            if fitness > gbest_fitness:
                gbest_fitness = fitness
                gbest_position = p.position.copy()

        best_history.append(gbest_fitness)
        _, nd = calc_normalized_damage(gbest_position)
        dmg_history.append(nd)

    return gbest_position.reshape(3,3), gbest_fitness, best_history, dmg_history

# HYBRID PSO-GA
def run_psoga():
    W_INERTIA = 0.9
    C1 = 2.5
    C2 = 2.5
    swarm = [Particle() for _ in range(SWARM_SIZE)]
    for _ in range(100):
        for p in swarm:
            r1 = np.random.rand(9)
            r2 = np.random.rand(9)
            gbest_position = swarm[0].position.copy()
            gbest_fitness = calc_fitness(gbest_position)
            for p in swarm:
               fit = calc_fitness(p.position)
               if fit > gbest_fitness:
                  gbest_fitness = fit
                  gbest_position = p.position.copy()
            p.velocity = (
                W_INERTIA * p.velocity
                + C1 * r1 * (p.pbest_position - p.position)
                + C2 * r2 * (gbest_position - p.position)
            )

            p.position = p.position + p.velocity
            p.position = np.round(p.position).astype(int)
            p.position = np.maximum(p.position, 0)

    for p in swarm:

        X = p.position.reshape(3,3)

        for r in range(3):

            total = np.sum(X[r])

            if total > Q[r]:
                factor = Q[r] / total
                X[r] = np.floor(X[r] * factor)
    p.position = X.flatten().astype(int)
    population = [p.position.copy() for p in swarm]
    best_solution = None
    best_fitness = -1
    best_history = []
    dmg_history = []

    for iteration in range(MAX_ITERATIONS):
        fitness_scores = [calc_fitness(ind)
                          for ind in population]
        best_idx = np.argmax(fitness_scores)
        current_best = population[best_idx]
        current_fitness = fitness_scores[best_idx]
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_solution = current_best.copy()

        best_history.append(best_fitness)
        _, nd = calc_normalized_damage(best_solution)
        dmg_history.append(nd)

        selected = random.choices(
            population,
            weights=[f + 0.01 for f in fitness_scores],
            k=POPULATION_SIZE
        )

        elite_count = 4
        elite_idx = np.argsort(fitness_scores)[-elite_count:]
        next_generation = [population[i].copy()
                   for i in elite_idx]

        while len(next_generation) < POPULATION_SIZE:
            p1, p2 = random.sample(selected, 2)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            for _ in range(2):
             c1= local_search(c1)
            for _ in range(2):
             c2= local_search(c2)
            next_generation.append(c1)
            if len(next_generation) < POPULATION_SIZE:
                next_generation.append(c2)

        population = next_generation

    return best_solution.reshape(3,3), best_fitness, best_history, dmg_history

# QPSO 
def run_qpso():
    c1, c2 = 2, 2
    r1, r2 = 0.6, 0.4
    b = 0.75
    THRESHOLD = 0.5
    BITS_PER_CELL = 4  
    CELLS = 9
    DIM = CELLS * BITS_PER_CELL  

    best_target_values = np.max(W[:, np.newaxis] * P * U, axis=1)
    MAX_THEORETICAL_DAMAGE = np.sum(Q * best_target_values)

    X = np.random.rand(SWARM_SIZE, DIM)
    PBest_X = np.copy(X)
    PBest_fitness = np.full(SWARM_SIZE, -np.inf)
    
    global_best_fitness = -np.inf
    global_best_X = np.zeros(DIM)
    global_best_matrix = None
    best_history = []
    dmg_history = []
    
    phi = (c1 * r1) / (c1 * r1 + c2 * r2)
    
    for generation in range(MAX_ITERATIONS):
        for i in range(SWARM_SIZE):
            binary_array = (X[i] > THRESHOLD).astype(int)
            
            x_integers = np.zeros(CELLS, dtype=int)
            for c in range(CELLS):
                bits = binary_array[c * BITS_PER_CELL : (c + 1) * BITS_PER_CELL]
                val = sum(bit * (2 ** idx) for idx, bit in enumerate(reversed(bits)))
                x_integers[c] = val

            X_matrix = x_integers.reshape((3, 3)) * U  
            raw_damage = np.sum(W[:, np.newaxis] * P * X_matrix)
            
            raw_fitness = raw_damage
            row_sums = np.sum(X_matrix, axis=1)
            
            for r in range(3):
                if row_sums[r] > Q[r]:
                    raw_fitness -= 10 * (row_sums[r] - Q[r])  
                    
            if raw_fitness > PBest_fitness[i]:
                PBest_fitness[i] = raw_fitness
                PBest_X[i] = np.copy(X[i])
                
            if raw_fitness > global_best_fitness:
                global_best_fitness = raw_fitness
                global_best_X = np.copy(X[i])
                global_best_matrix = np.copy(X_matrix)
                
        best_history.append(max(0.0, global_best_fitness) / MAX_THEORETICAL_DAMAGE)
        
        if global_best_matrix is not None:
            _, nd = calc_normalized_damage(global_best_matrix)
        else:
            nd = 0
        dmg_history.append(nd)
        
        mbest = np.mean(PBest_X, axis=0)
        
        for i in range(SWARM_SIZE):
            p = phi * PBest_X[i] + (1 - phi) * global_best_X
            u = np.random.rand(DIM)
            u = np.maximum(u, 1e-10) 
            L = b * np.abs(mbest - X[i])
            sign = np.where(np.random.rand(DIM) > 0.5, 1, -1)
            X[i] = p + sign * L * np.log(1.0 / u)

    return global_best_matrix, max(0.0, global_best_fitness) / MAX_THEORETICAL_DAMAGE, best_history, dmg_history

# QIGA
def run_qiga():
    BITS_PER_CELL = 4  
    CELLS = 9
    DIM = CELLS * BITS_PER_CELL
    MUTATION_RATE = 0.05
    SLOW_THETA_STEP = 0.02 * np.pi

    best_target_values = np.max(W[:, np.newaxis] * P * U, axis=1)
    MAX_THEORETICAL_DAMAGE = np.sum(Q * best_target_values)

    pop_angles = np.full((POPULATION_SIZE, DIM), np.pi / 4)
    
    global_best_fitness = -np.inf
    global_best_x = None
    global_best_matrix = None
    best_history = []
    dmg_history = []
    
    for generation in range(MAX_ITERATIONS):
        pop_measurements = np.zeros((POPULATION_SIZE, DIM), dtype=int)
        
        for i in range(POPULATION_SIZE):
            probs = np.sin(pop_angles[i]) ** 2
            binary_array = (np.random.rand(DIM) < probs).astype(int)
            pop_measurements[i] = binary_array
            
            x_integers = np.zeros(CELLS, dtype=int)
            for c in range(CELLS):
                bits = binary_array[c * BITS_PER_CELL : (c + 1) * BITS_PER_CELL]
                val = sum(bit * (2 ** idx) for idx, bit in enumerate(reversed(bits)))
                x_integers[c] = val

            X_matrix = x_integers.reshape((3, 3)) * U 
            raw_damage = np.sum(W[:, np.newaxis] * P * X_matrix)
            
            penalty_factor = 1.0
            row_sums = np.sum(X_matrix, axis=1)
            for r in range(3):
                if row_sums[r] > Q[r]:
                    penalty_factor *= 0.5 ** (row_sums[r] - Q[r]) 
                    
            fit_val = raw_damage * penalty_factor
            
            if fit_val > global_best_fitness:
                global_best_fitness = fit_val
                global_best_x = binary_array.copy()
                global_best_matrix = X_matrix.copy()
        
        best_history.append(max(0.0, global_best_fitness) / MAX_THEORETICAL_DAMAGE)
        
        if global_best_matrix is not None:
            _, nd = calc_normalized_damage(global_best_matrix)
        else:
            nd = 0
        dmg_history.append(nd)
        
        for i in range(POPULATION_SIZE):
            for qubit in range(DIM):
                x_val = pop_measurements[i, qubit]
                best_val = global_best_x[qubit]
                
                if x_val == 0 and best_val == 1:
                    pop_angles[i, qubit] = min(pop_angles[i, qubit] + SLOW_THETA_STEP, np.pi / 2)
                elif x_val == 1 and best_val == 0:
                    pop_angles[i, qubit] = max(pop_angles[i, qubit] - SLOW_THETA_STEP, 0.0)

                if np.random.rand() < MUTATION_RATE:
                    mutation_shift = np.random.uniform(-0.05 * np.pi, 0.05 * np.pi)
                    pop_angles[i, qubit] = np.clip(pop_angles[i, qubit] + mutation_shift, 0.0, np.pi / 2)

    return global_best_matrix, max(0.0, global_best_fitness) / MAX_THEORETICAL_DAMAGE, best_history, dmg_history


# RESULT
ga_X, ga_fit, ga_hist, ga_dmg_hist = run_ga()
pso_X, pso_fit, pso_hist, pso_dmg_hist = run_pso()
gapso_X, gapso_fit, gapso_hist, gapso_dmg_hist = run_gapso()
psoga_X, psoga_fit, psoga_hist, psoga_dmg_hist = run_psoga()
qiga_X, qiga_fit, qiga_hist, qiga_dmg_hist = run_qiga()
qpso_X, qpso_fit, qpso_hist, qpso_dmg_hist = run_qpso()

ga_damage, ga_nd = calc_normalized_damage(ga_X)
pso_damage, pso_nd = calc_normalized_damage(pso_X)
gapso_damage, gapso_nd = calc_normalized_damage(gapso_X)
psoga_damage, psoga_nd = calc_normalized_damage(psoga_X)
qpso_damage, qpso_nd = calc_normalized_damage(qpso_X)
qiga_damage, qiga_nd = calc_normalized_damage(qiga_X)

print("\nQPSO")
print(qiga_X)
print(f"Fitness Score: {qiga_fit:.4f}")
print(f"Normalized Damage: {qiga_nd:.4f}")

print("\nQIGA")
print(qpso_X)
print(f"Fitness Score: {qpso_fit:.4f}")
print(f"Normalized Damage: {qpso_nd:.4f}")

print("\nHYBRID PSO-GA")
print(psoga_X)
print(f"Fitness Score: {psoga_fit:.4f}")
print(f"Normalized Damage: {psoga_nd:.4f}")

print("\nHYBRID GA-PSO")
print(gapso_X)
print(f"Fitness Score: {gapso_fit:.4f}")
print(f"Normalized Damage: {gapso_nd:.4f}")

print("\nPSO")
print(pso_X)
print(f"Fitness Score: {pso_fit:.4f}")
print(f"Normalized Damage: {pso_nd:.4f}")

print("\nGA")
print(ga_X)
print(f"Fitness Score: {ga_fit:.4f}")
print(f"Normalized Damage: {ga_nd:.4f}")

# GRAPHING BOTH HISTORIES
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Fitness
ax1.plot(qpso_hist, label='QPSO', linewidth=2)
ax1.plot(qiga_hist, label='QIGA', linewidth=2)
ax1.plot(psoga_hist, label='PSO-GA', linewidth=2)
ax1.plot(gapso_hist, label='GA-PSO', linewidth=2)
ax1.plot(pso_hist, label='PSO', linewidth=2)
ax1.plot(ga_hist, label='GA', linewidth=2)
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Fitness")
ax1.set_title("Fitness Comparison of Algorithms")
ax1.legend(loc="lower right")
ax1.grid(True, linestyle=':', alpha=0.7)

# Plot 2: Damage

algorithms = ['GA', 'PSO', 'GAPSO', 'PSOGA', 'QIGA', 'QPSO']
damage_values = [
    ga_nd,
    qpso_nd,
    gapso_nd,
    psoga_nd,  
    pso_nd,
    qiga_nd
]

bars = ax2.bar(
    algorithms,
    damage_values,
    width=0.8
)

ax2.set_ylim(0, max(damage_values) + 0.1)
ax2.set_ylabel('Normalized Damage')
ax2.set_title('Overall Damage')
ax2.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()