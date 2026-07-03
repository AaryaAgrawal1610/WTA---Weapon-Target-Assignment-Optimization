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

# PSO Parameters 
SWARM_SIZE = 20
MAX_ITERATIONS = 60
W_INERTIA = 0.7   
C1 = 2          
C2 = 2          

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

def initialize_particle_position():
    chrom = []
    for i in range(3):
        alloc = [0,0,0]
        for _ in range(Q[i]):      
            valid_targets = [j for j in range(3) if U[i][j]==1]
            target = random.choice(valid_targets)
            alloc[target] += 1
        chrom.extend(alloc)
    return np.array(chrom)

class Particle:
    def __init__(self):
        self.position = initialize_particle_position()
        self.velocity = np.random.uniform(-1, 1, 9)
        self.pbest_position = self.position.copy()
        self.pbest_fitness = calc_fitness(self.position)

# EXECUTE PSO
swarm = [Particle() for _ in range(SWARM_SIZE)]
best_fitness_history = []
gbest_position = None
gbest_fitness = -1.0

# Initial assessment for global best
for p in swarm:
    if p.pbest_fitness > gbest_fitness:
        gbest_fitness = p.pbest_fitness
        gbest_position = p.pbest_position.copy()

for iteration in range(MAX_ITERATIONS):
    for p in swarm:
        # Update Velocity
        r1 = np.random.rand(9)
        r2 = np.random.rand(9)
        p.velocity = (W_INERTIA * p.velocity + 
                      C1 * r1 * (p.pbest_position - p.position) + 
                      C2 * r2 * (gbest_position - p.position))
        
        # Update Position
        new_position = p.position + p.velocity
        
        # Handle discrete space & constraints
        new_position = np.round(new_position).astype(int)
        new_position = np.maximum(new_position, 0) 
        
        for i in range(9):
            row = i // 3
            col = i % 3
            if U[row][col] == 0:
                new_position[i] = 0
                
        p.position = new_position
        
        # Evaluate Fitness
        current_fitness = calc_fitness(p.position)
        
        # Update Personal and Global Bestsx
        for p in swarm:
         if current_fitness > p.pbest_fitness:
              p.pbest_fitness = current_fitness
              p.pbest_position = p.position.copy()

         if p.pbest_fitness > gbest_fitness:
             gbest_fitness = p.pbest_fitness
             gbest_position = p.pbest_position.copy()

    best_fitness_history.append(gbest_fitness)
    
    # Output of every iteration
    num_targets = 3 
    X_best = np.array(gbest_position).reshape(3, 3)
    weapons_used = [int(np.sum(X_best[i])) for i in range(3)]
    damage_per_target = np.sum(P * X_best, axis=0) 
    overall_damage = sum(damage_per_target) / num_targets
    particle_matrix = p.position.reshape(3, 3)
    max_damage = 0
    for i in range(3):
     max_damage += Q[i] * np.max(P[i])
    normalized_damage = overall_damage / max_damage
    
    print(f"Iteration {iteration + 1}:")
    print(f"Last Particle Matrix: : \n{particle_matrix}")
    print("Velocity:")
    print(np.array2string(
    p.velocity.reshape(3, 3),
    formatter={'float_kind': lambda x: f"{x:.2f}"}
    ))
    print(f"PBest = \n{p.pbest_position.reshape(3, 3)}")
    print(f"GBest = \n{gbest_position.reshape(3, 3)}")
    print(f"Overall Damage: {normalized_damage:.2f}")
    print(f"Fitness Score : {gbest_fitness:.4f}")
    print("-" * 75)

optimal_X = np.array(gbest_position).reshape(3, 3)

# VISUALIZATION
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Stage 1: PSO Convergence
ax1.plot(range(MAX_ITERATIONS), best_fitness_history, color='r', linewidth=2)
ax1.set_title("Stage 1: PSO Convergence over Iterations")
ax1.set_xlabel("Iterations")
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
ax2.set_title("Stage 2: Simulated Defensive Positions (Contour)")
ax2.set_xlabel("X Coordinates")
ax2.set_ylabel("Y Coordinates")

plt.tight_layout()
plt.show()

# comparison
print("\nRESULTS COMPARISON")
print("Paper's Stated Optimal X:")
print(np.array([[1, 2, 2], [4, 0, 2], [0, 3, 1]]))
print("Paper's Stated Fitness (F): 0.6456\n")
print("PSO Computed Optimal X:")
print(optimal_X)
print(f"PSO Computed Fitness: {gbest_fitness:.4f}")