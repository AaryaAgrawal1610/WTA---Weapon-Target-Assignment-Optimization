import numpy as np
import matplotlib.pyplot as plt
import random

# GIVEN DATA
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

# PARAMETERS
MAX_ITERATIONS = 60

# PSO
SWARM_SIZE = 20
W_INERTIA = 0.7
C1 = 2
C2 = 2

# GA
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

def generate_random_allocation():
    allocation = []
    for i in range(3):
        row = [0, 0, 0]
        weapons = random.randint(0, Q[i])
        for _ in range(weapons):
            target = random.randint(0, 2)
            if U[i][target] == 1:
                row[target] += 1
        allocation.extend(row)
    return np.array(allocation)

# GA PHASE
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
        self.position = generate_random_allocation()
        self.velocity = np.random.uniform(-1, 1, 9)
        self.pbest_position = self.position.copy()
        self.pbest_fitness = calc_fitness(self.position)

# HYBRID GA-PSO
def run_hybrid_gapso():
    swarm = [Particle() for _ in range(SWARM_SIZE)]
    best_fitness_history = []

    # INITIAL GLOBAL BEST
    gbest_position = swarm[0].position.copy()
    gbest_fitness = calc_fitness(gbest_position)

    for particle in swarm:
        fit = calc_fitness(particle.position)
        if fit > gbest_fitness:
            gbest_fitness = fit
            gbest_position = particle.position.copy()

    for iteration in range(MAX_ITERATIONS):
        print("\n" + "=" * 90)
        print(f"ITERATION {iteration + 1}")
        print("=" * 90)

        # PSO PHASE
        for idx, particle in enumerate(swarm):
            r1 = np.random.rand(9)
            r2 = np.random.rand(9)

            # Velocity Update
            particle.velocity = (
                W_INERTIA * particle.velocity
                + C1 * r1 * (particle.pbest_position - particle.position)
                + C2 * r2 * (gbest_position - particle.position)
            )

            # Position Update
            particle.position = particle.position + particle.velocity
            particle.position = np.round(particle.position).astype(int)
            particle.position = np.maximum(particle.position, 0)

            # Respect U matrix
            for j in range(9):
                row = j // 3
                col = j % 3
                if U[row][col] == 0:
                    particle.position[j] = 0

            # Respect weapon quantity constraints
            X = particle.position.reshape(3, 3)
            for r in range(3):
                total = np.sum(X[r])
                if total > Q[r]:
                    factor = Q[r] / total
                    X[r] = np.floor(X[r] * factor)
            particle.position = X.flatten().astype(int)

        # GA PHASE
        random.shuffle(swarm)
        for i in range(0, SWARM_SIZE, 2):
            parent1 = swarm[i].position
            parent2 = swarm[i + 1].position
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            swarm[i].position = child1
            swarm[i + 1].position = child2

        # EVALUATION
        for idx, particle in enumerate(swarm):
            fitness = calc_fitness(particle.position)
            if fitness > particle.pbest_fitness:
                particle.pbest_fitness = fitness
                particle.pbest_position = particle.position.copy()
            if fitness > gbest_fitness:
                gbest_fitness = fitness
                gbest_position = particle.position.copy()

            # RESULT (Moved variable calculation up to avoid reference error)
            damage = np.sum(P * particle.position.reshape(3, 3), axis=0)
            overall_damage = np.sum(damage)
            
            max_damage = 0
            for i in range(3):
                max_damage += Q[i] * np.max(P[i])
            normalized_damage = overall_damage / max_damage
            
            print(f"\nParticle {idx + 1}")
            print("-" * 60)
            print("Particle Matrix:")
            print(particle.position.reshape(3, 3))
            print("\nPosition Matrix:")
            print(particle.position.reshape(3, 3))
            print("\nVelocity Matrix:")
            print(particle.velocity.reshape(3, 3))
            print("\nPBest Matrix:")
            print(particle.pbest_position.reshape(3, 3))
            print("\nGBest Matrix:")
            print(gbest_position.reshape(3, 3))
            print(f"\nOverall Damage: {normalized_damage:.4f}")
            print(f"Fitness Score: {fitness:.4f}")

        best_fitness_history.append(gbest_fitness)

        print("\n" + "#" * 90)
        print("BEST SOLUTION SO FAR")
        print("#" * 90)
        print("GBest Position:")
        print(gbest_position.reshape(3, 3))
        
        # Fixed escape sequence error (\G -> \nG)
        print(f"\nGBest Fitness: {gbest_fitness:.4f}")

    return gbest_position.reshape(3, 3), gbest_fitness, best_fitness_history

# Removed the duplicate execution step 
hybrid_optimal_X, hybrid_best_fitness, hybrid_history = run_hybrid_gapso()

# COMPARISON
paper_optimal_X = np.array([[1, 2, 2], [4, 0, 2], [0, 3, 1]])
paper_fitness = 0.6456
print("\n[Paper's Stated Optimal]")
print(f"X:\n{paper_optimal_X}")
print(f"Fitness: {paper_fitness}")
print("\n[Hybrid GA-PSO Result]")
print(f"X:\n{hybrid_optimal_X}")
print(f"\nFinal Fitness: {hybrid_best_fitness:.4f}")

# VISUALIZATION
def plot_contour(ax, optimal_X, title, colormap):
    x = np.linspace(0, 50, 100)
    y = np.linspace(0, 50, 100)
    X_grid, Y_grid = np.meshgrid(x, y)
    Z = np.zeros_like(X_grid)
    
    random.seed(42) 
    # Fixed range to 3 to cover the full 3x3 matrix layout
    for i in range(3):
        for j in range(3):
            if optimal_X[i][j] > 0:
                cx, cy = random.uniform(10, 40), random.uniform(10, 40)
                Z += optimal_X[i][j] * np.exp(-((X_grid - cx)**2 + (Y_grid - cy)**2) / 20)

    ax.contour(X_grid, Y_grid, Z, levels=10, cmap=colormap, linewidths=1.5)
    ax.set_title(title)
    ax.set_xlabel("X Coordinates")
    ax.set_ylabel("Y Coordinates")

fig, axs = plt.subplots(1, 2, figsize=(18, 5))

# Plot 1: Stage 1: Hybrid GA-PSO Convergence (Added label parameter)
axs[0].plot(range(MAX_ITERATIONS), hybrid_history, linewidth=2, label="Best Fitness")
axs[0].set_title("Convergence")
axs[0].set_xlabel("Iterations")
axs[0].set_ylabel("Fitness Score")
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.6)

# Plot 2: Stage 2: Defensive Distribution
plot_contour(axs[1], hybrid_optimal_X, "Hybrid GA-PSO Allocations", 'Greens')

plt.tight_layout()
plt.show()