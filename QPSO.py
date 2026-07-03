import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.preprocessing import MinMaxScaler

c1 = 2
c2 = 2
r1 = 0.6
r2 = 0.4
b = 0.75
THRESHOLD = 0.5

POP_SIZE = 50
ITERATIONS = 60

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

w_reshaped_global = W[:, np.newaxis]
w_reshaped_global = W[:, np.newaxis]

best_target_values = np.max(w_reshaped_global * P * U, axis=1)
MAX_THEORETICAL_DAMAGE = np.sum(Q * best_target_values)

BITS_PER_CELL = 4 
CELLS = 9
DIM = CELLS * BITS_PER_CELL  

def calculate_fitness(binary_array):
    x_integers = np.zeros(CELLS, dtype=int)
    for i in range(CELLS):
        bits = binary_array[i * BITS_PER_CELL : (i + 1) * BITS_PER_CELL]
        
        val = 0
        for idx, bit in enumerate(reversed(bits)):
            val += bit * (2 ** idx)
        
        x_integers[i] = val

    X_matrix = x_integers.reshape((3, 3))
    X_matrix = X_matrix * U  
    
    w_reshaped = W[:, np.newaxis] 
    raw_damage = np.sum(w_reshaped * P * X_matrix)
    
    raw_fitness = raw_damage
    row_sums = np.sum(X_matrix, axis=1)
    
    for i in range(3):
        if row_sums[i] > Q[i]:
            raw_fitness -= 10 * (row_sums[i] - Q[i])  
            
    return raw_fitness, raw_damage, X_matrix


def run_qpso():
    X = np.random.rand(POP_SIZE, DIM)
    PBest_X = np.copy(X)
    PBest_fitness = np.full(POP_SIZE, -np.inf)
    
    global_best_fitness = -np.inf
    global_best_X = np.zeros(DIM)
    global_best_matrix = None
    global_best_binary = None
    
    best_fitness_history = []
    avg_fitness_history = []
    best_damage_history = []
    phi = (c1 * r1) / (c1 * r1 + c2 * r2)
    
    for generation in range(ITERATIONS):
        pop_fitness = np.zeros(POP_SIZE)
        
        gen_best_fitness = -np.inf
        gen_best_damage = 0.0
        gen_best_matrix = None
        
        for i in range(POP_SIZE):
            binary_array = (X[i] > THRESHOLD).astype(int)
            
            fit_val, dmg_val, x_mat = calculate_fitness(binary_array)
            pop_fitness[i] = fit_val
            
            if fit_val > PBest_fitness[i]:
                PBest_fitness[i] = fit_val
                PBest_X[i] = np.copy(X[i])
                
            if fit_val > gen_best_fitness:
                gen_best_fitness = fit_val
                gen_best_damage = dmg_val
                gen_best_matrix = np.copy(x_mat)
                
            if fit_val > global_best_fitness:
                global_best_fitness = fit_val
                global_best_X = np.copy(X[i])
                global_best_binary = np.copy(binary_array)
                global_best_matrix = np.copy(x_mat)
                
        avg_pop_fitness = np.mean(pop_fitness)
        best_fitness_history.append(global_best_fitness)
        avg_fitness_history.append(avg_pop_fitness)
        best_damage_history.append(gen_best_damage)
        
        mbest = np.mean(PBest_X, axis=0)
        
        for i in range(POP_SIZE):
            p = phi * PBest_X[i] + (1 - phi) * global_best_X
            u = np.random.rand(DIM)
            u = np.maximum(u, 1e-10) 
            L = b * np.abs(mbest - X[i])
            sign = np.where(np.random.rand(DIM) > 0.5, 1, -1)
            
            X[i] = p + sign * L * np.log(1.0 / u)

        normalized_damage = gen_best_damage / MAX_THEORETICAL_DAMAGE
        normalized_fitness = gen_best_fitness / MAX_THEORETICAL_DAMAGE
        
        
        print(f"Iteration {generation+1}:")
        _, _, mbest_matrix = calculate_fitness((mbest > THRESHOLD).astype(int))
        _, _, pbest_matrix = calculate_fitness((PBest_X[-1] > THRESHOLD).astype(int)) 
        
        print("Generation Best Matrix: ")
        for row in gen_best_matrix:
            print(row)
        print(f"mBest =\n{mbest_matrix}")
        print(f"PBest (Last Particle) =\n{pbest_matrix}")
        print(f"GBest =\n{global_best_matrix}")
        print(f"Overall Damage   : {normalized_damage:.4f}")
        print(f"Fitness Score    : {normalized_fitness:.4f}")
        print("-" * 75)

    return global_best_fitness, global_best_matrix, best_fitness_history, avg_fitness_history, best_damage_history

# Execution
if __name__ == "__main__":
    best_fit, best_sol, best_hist, avg_hist, dmg_hist = run_qpso()
    
    final_norm_fit = best_fit / MAX_THEORETICAL_DAMAGE
    print("\nRESULTS COMPARISON")
    print("Paper's Stated Optimal X:")
    print(np.array([[1, 2, 2], [4, 0, 2], [0, 3, 1]]))
    print("Paper's Stated Fitness (F): 0.6456\n")
    print("QPSO Computed Optimal X:")
    print(best_sol)
    print(f"QPSO Fitness : {final_norm_fit:.4f}")
    
    # Plot 1: Fitness Graph
    plt.subplot(1, 2, 1)
    plt.plot(range(1, ITERATIONS + 1), best_hist, color='blue', linewidth=2)
    plt.title('Fitness', fontsize=14)
    plt.xlabel('Generations', fontsize=12)
    plt.ylabel('Fitness Value', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Plot 2: Damage Graph
    plt.subplot(1, 2, 2)
    plt.hist( dmg_hist, bins=10, color='red', edgecolor='black', alpha=0.7)
    plt.title('Damage Distribution', fontsize=14)
    plt.xlabel('Damage Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()