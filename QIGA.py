import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.preprocessing import MinMaxScaler

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

THETA_STEP = 0.05 * np.pi

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
    
    penalty_factor = 1.0
    for i in range(3):
        if row_sums[i] > Q[i]:
            penalty_factor *= 0.5 ** (row_sums[i] - Q[i]) 
            
    raw_fitness = raw_damage * penalty_factor
    
    return raw_fitness, raw_damage, X_matrix

MUTATION_RATE = 0.05  

def run_qiga():
    scaler = MinMaxScaler()
    pop_angles = np.full((POP_SIZE, DIM), np.pi / 4)
    
    global_best_fitness = -np.inf
    global_best_x = None
    global_best_matrix = None
    
    best_fitness_history = []
    avg_fitness_history = []
    best_damage_history = []
    
    for generation in range(ITERATIONS):
        pop_fitness = np.zeros(POP_SIZE)
        pop_measurements = np.zeros((POP_SIZE, DIM), dtype=int)
        
        gen_best_fitness = -np.inf
        gen_best_damage = 0.0
        gen_best_matrix = None
        
        for i in range(POP_SIZE):
            probs = np.sin(pop_angles[i]) ** 2
            binary_array = (np.random.rand(DIM) < probs).astype(int)
            pop_measurements[i] = binary_array
            
            fit_val, dmg_val, x_mat = calculate_fitness(binary_array)
            pop_fitness[i] = fit_val
            
            if fit_val > gen_best_fitness:
                gen_best_fitness = fit_val
                gen_best_damage = dmg_val
                gen_best_matrix = x_mat.copy()
            
            if fit_val > global_best_fitness:
                global_best_fitness = fit_val
                global_best_x = binary_array.copy()
                global_best_matrix = x_mat.copy()
        
        avg_pop_fitness = np.mean(pop_fitness)
        
        best_fitness_history.append(global_best_fitness)
        avg_fitness_history.append(avg_pop_fitness)
        best_damage_history.append(gen_best_damage)
        
        for i in range(POP_SIZE):
            for qubit in range(DIM):
                x_val = pop_measurements[i, qubit]
                best_val = global_best_x[qubit]
                
                SLOW_THETA_STEP = 0.02 * np.pi 
                
                if x_val == 0 and best_val == 1:
                    pop_angles[i, qubit] = min(pop_angles[i, qubit] + SLOW_THETA_STEP, np.pi / 2)
                elif x_val == 1 and best_val == 0:
                    pop_angles[i, qubit] = max(pop_angles[i, qubit] - SLOW_THETA_STEP, 0.0)

                if np.random.rand() < MUTATION_RATE:
                    mutation_shift = np.random.uniform(-0.05 * np.pi, 0.05 * np.pi)
                    pop_angles[i, qubit] = np.clip(pop_angles[i, qubit] + mutation_shift, 0.0, np.pi / 2)

        normalized_damage = gen_best_damage / MAX_THEORETICAL_DAMAGE
        normalized_fitness = gen_best_fitness / MAX_THEORETICAL_DAMAGE
        normalized_avg = avg_pop_fitness / MAX_THEORETICAL_DAMAGE
        
        print(f"Iteration {generation+1}:")
        print("Weapon Assignment Matrix:")
        for row in gen_best_matrix:
            print(row)
        print(f"Overall Damage   : {normalized_damage:.4f}")
        print(f"Fitness Score    : {normalized_fitness:.4f}")
        print("-" * 75)

    return global_best_fitness, global_best_matrix, best_fitness_history, avg_fitness_history, best_damage_history

# Execution
if __name__ == "__main__":
    best_fit, best_sol, best_hist, avg_hist, dmg_hist = run_qiga()
    
    final_norm_fit = best_fit / MAX_THEORETICAL_DAMAGE
    print("\nRESULTS COMPARISON")
    print("Paper's Stated Optimal X:")
    print(np.array([[1, 2, 2], [4, 0, 2], [0, 3, 1]]))
    print("Paper's Stated Fitness (F): 0.6456\n")
    print("QIGA Computed Optimal X:")
    print(best_sol)
    print(f"QIGA Fitness : {final_norm_fit:.4f}")
    
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