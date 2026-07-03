import numpy as np
import matplotlib.pyplot as plt
import random

# Given data
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

MAX_ITERATIONS = 60

# PSO Parameters 
SWARM_SIZE = 20
W_INERTIA = 0.7   
C1 = 2          
C2 = 2          

# GA Parameters 
MUTATION_RATE = 0.01

# Functions

def calc_fitness(chromosome):
    X = np.array(chromosome).reshape(3, 3)
    for i in range(3):
        if np.sum(U[i] * X[i]) > Q[i]:
            return 0.0  # Invalid 
        
    raw_damage = np.sum(W.reshape(3, 1) * P * X)
    max_damage = np.sum(W.reshape(3, 1) * P * np.array(Q).reshape(3, 1))
    fitness_value = raw_damage / max_damage 
    return fitness_value

def generate_random_allocation():
    chrom = []
    for i in range(3):
        alloc = [0, 0, 0]
        total_weapons = random.randint(0, Q[i])
        for _ in range(total_weapons):
            target = random.choice([0, 1, 2])
            if U[i][target] == 1:  
                alloc[target] += 1
        chrom.extend(alloc)
    return np.array(chrom)

def crossover(parent1, parent2):
    # Convert to lists if numpy arrays
    p1 = list(parent1) if isinstance(parent1, np.ndarray) else list(parent1)
    p2 = list(parent2) if isinstance(parent2, np.ndarray) else list(parent2)
    pt = random.randint(1, 8)
    child1 = p1[:pt] + p2[pt:]
    child2 = p2[:pt] + p1[pt:]
    return child1, child2

def mutate(chromosome):
    # Convert to list for mutation
    chrom = list(chromosome) if isinstance(chromosome, np.ndarray) else list(chromosome)
    if random.random() < MUTATION_RATE:
        idx = random.randint(0, 8)
        row, col = idx // 3, idx % 3
        if U[row][col] == 1:
            chrom[idx] = random.randint(0, Q[row])
    return chrom

class Particle:
    def __init__(self):
        self.position = generate_random_allocation()
        self.velocity = np.random.uniform(-1, 1, 9)
        self.pbest_position = self.position.copy()
        self.pbest_fitness = calc_fitness(self.position)

# Implementation
def run_hybrid_gapso():
    swarm = [Particle() for _ in range(SWARM_SIZE)]
    best_fitness_history = []
    gbest_position = None
    gbest_fitness = -1.0

    for p in swarm:
        if p.pbest_fitness > gbest_fitness:
            gbest_fitness = p.pbest_fitness
            gbest_position = p.pbest_position.copy()

    for iteration in range(MAX_ITERATIONS):
        # 1. PSO Phase: Update Velocity & Position
        for p in swarm:
            r1, r2 = np.random.rand(9), np.random.rand(9)
            p.velocity = (W_INERTIA * p.velocity + 
                          C1 * r1 * (p.pbest_position - p.position) + 
                          C2 * r2 * (gbest_position - p.position))
            p.position = p.position + p.velocity

        # 2. GA Phase: Crossover and Mutation on Swarm Positions
        positions = [p.position.tolist() for p in swarm]
        next_positions = [None] * SWARM_SIZE
        
        # Shuffle indices 
        indices = list(range(SWARM_SIZE))
        random.shuffle(indices)
        
        for i in range(0, SWARM_SIZE, 2):
            p1_idx, p2_idx = indices[i], indices[i+1]
            c1, c2 = crossover(positions[p1_idx], positions[p2_idx])
            next_positions[p1_idx] = c1
            next_positions[p2_idx] = c2

        # Apply Mutation
        for i, p in enumerate(swarm):
            mutated_pos = mutate(next_positions[i])
            
            new_position = np.array(mutated_pos)
            new_position = np.round(new_position).astype(int)
            new_position = np.maximum(new_position, 0)
            
            for j in range(9):
                row, col = j // 3, j % 3
                if U[row][col] == 0:
                    new_position[j] = 0
                    
            p.position = new_position
            current_fitness = calc_fitness(p.position)
            
            # 3. Update PBest and GBest
            if current_fitness > p.pbest_fitness:
                p.pbest_fitness = current_fitness
                p.pbest_position = p.position.copy()
                
                if current_fitness > gbest_fitness:
                    gbest_fitness = current_fitness
                    gbest_position = p.position.copy()

        best_fitness_history.append(gbest_fitness)
        
        # OUTPUT FOR EACH ITERATION
        print("\n" + "=" * 90)
        print(f"ITERATION {iteration + 1}")
        print("=" * 90)

        for idx, particle in enumerate(swarm):
            fitness = calc_fitness(particle.position)

            # Calculate damage for this particle
            damage = np.sum(P * particle.position.reshape(3, 3), axis=0)
            overall_damage = np.sum(damage)
            
            # Calculate max damage for normalization
            max_damage = 0
            for i in range(3):
                max_damage += Q[i] * np.max(P[i])
            normalized_damage = overall_damage / max_damage if max_damage > 0 else 0

            print(f"\nParticle {idx + 1}")
            print("-" * 60)

            print("Position Matrix:")
            print(particle.position.reshape(3, 3))

            print("\nVelocity Matrix:")
            print(np.round(particle.velocity.reshape(3, 3), 4))

            print("\nPBest Matrix:")
            print(particle.pbest_position.reshape(3, 3))

            print("\nGBest Matrix:")
            print(gbest_position.reshape(3, 3))

            print(f"\nOverall Damage: {normalized_damage:.4f}")
            print(f"Fitness Score: {fitness:.4f}")

        print("\n" + "#" * 90)
        print("BEST SOLUTION SO FAR")
        print("#" * 90)

        print("GBest Position:")
        print(gbest_position.reshape(3, 3))

        print(f"\nGBest Fitness: {gbest_fitness:.4f}")

    return np.array(gbest_position).reshape(3, 3), gbest_fitness, best_fitness_history

# COMPARISON
hybrid_optimal_X, hybrid_best_fitness, hybrid_history = run_hybrid_gapso()

paper_optimal_X = np.array([[1, 2, 2], [4, 0, 2], [0, 3, 1]])
paper_fitness = 0.6456

print("\n[Paper's Stated Optimal]")
print(f"X:\n{paper_optimal_X}")
print(f"Fitness: {paper_fitness}")

print("\n[Hybrid GA-PSO Result]")
print(f"X:\n{hybrid_optimal_X}")
print(f"Fitness: {hybrid_best_fitness:.4f}")

# VISUALIZATION
def plot_contour(ax, optimal_X, title, colormap):
    x = np.linspace(0, 50, 100)
    y = np.linspace(0, 50, 100)
    X_grid, Y_grid = np.meshgrid(x, y)
    Z = np.zeros_like(X_grid)
    
    random.seed(42) 
    for i in range(2):
        for j in range(2):
            if optimal_X[i][j] > 0:
                cx, cy = random.uniform(10, 40), random.uniform(10, 40)
                Z += optimal_X[i][j] * np.exp(-((X_grid - cx)**2 + (Y_grid - cy)**2) / 20)

    ax.contour(X_grid, Y_grid, Z, levels=10, cmap=colormap, linewidths=1.5)
    ax.set_title(title)
    ax.set_xlabel("X Coordinates")
    ax.set_ylabel("Y Coordinates")

fig, axs = plt.subplots(1, 2, figsize=(18, 5))

# Plot 1: Hybrid GA-PSO Convergence
axs[0].plot(range(len(hybrid_history)),
            hybrid_history,
            linewidth=2,
            label='Hybrid GA-PSO')

axs[0].set_title("Convergence")
axs[0].set_xlabel("Iterations")
axs[0].set_ylabel("Fitness Score")
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.6)

# Plot 2: Defensive Distribution
plot_contour(axs[1], hybrid_optimal_X, "Hybrid PSO-GA Allocations", 'Greens')

plt.tight_layout()
plt.show()
