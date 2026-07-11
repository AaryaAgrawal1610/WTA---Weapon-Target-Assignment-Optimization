import numpy as np
import pandas as pd
import copy
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import random
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.preprocessing import MinMaxScaler

class WTA_Scenario:
    def __init__(self, w, t):
        self.W = w
        self.T = t
        self.u = np.random.randint(10, 100, size=t)
        self.max_possible_damage = np.sum(self.u)
        self.p = np.random.uniform(0.1, 0.9, size=(w, t))
        self.q = 1.0 - self.p

    def calc_damage(self, assignment):
        expected_survival_value = 0
        for j in range(self.T):
            survival_prob = 1.0
            for i in range(self.W):
                if assignment[i] == j:
                    survival_prob *= self.q[i, j]
            expected_survival_value += self.u[j] * survival_prob
        return ((self.max_possible_damage - expected_survival_value) / self.max_possible_damage) * 100

    def calc_fitness(self, assignment):
        return (self.calc_damage(assignment) / 100) ** 0.5

def init_pop(pop_size, W, T):
    return [np.random.randint(0, T, size=W) for _ in range(pop_size)]

def algo_GA(scenario, iterations, pop_size):
    pop = init_pop(pop_size, scenario.W, scenario.T)
    best_history = []
    gbest_val = -1
    gbest = pop[0]
    for _ in range(iterations):
        fitness = [scenario.calc_fitness(ind) for ind in pop]
        current_best_idx = np.argmax(fitness)
        if fitness[current_best_idx] > gbest_val:
            gbest_val = fitness[current_best_idx]
            gbest = copy.deepcopy(pop[current_best_idx])
        best_history.append(gbest_val)
        new_pop = []
        for _ in range(pop_size):
            i1, i2 = np.random.choice(pop_size, 2, replace=False)
            parent1 = pop[i1] if fitness[i1] > fitness[i2] else pop[i2]
            i3, i4 = np.random.choice(pop_size, 2, replace=False)
            parent2 = pop[i3] if fitness[i3] > fitness[i4] else pop[i4]
            child = copy.deepcopy(parent1)
            mask = np.random.rand(scenario.W) < 0.5
            child[mask] = parent2[mask]
            if np.random.rand() < 0.1:
                mut_idx = np.random.randint(0, scenario.W)
                child[mut_idx] = np.random.randint(0, scenario.T)
            new_pop.append(child)
        pop = new_pop
    return best_history, scenario.calc_damage(gbest)

def algo_PSO(scenario, iterations, pop_size):
    pop = init_pop(pop_size, scenario.W, scenario.T)
    velocities = [np.zeros(scenario.W) for _ in range(pop_size)]
    pbests = copy.deepcopy(pop)
    pbest_vals = [scenario.calc_fitness(ind) for ind in pop]
    gbest = copy.deepcopy(pop[np.argmax(pbest_vals)])
    gbest_val = np.max(pbest_vals)
    best_history = []
    w, c1, c2 = 0.7, 1.5, 1.5
    for _ in range(iterations):
        for i in range(pop_size):
            r1, r2 = np.random.rand(scenario.W), np.random.rand(scenario.W)
            velocities[i] = w * velocities[i] + c1 * r1 * (pbests[i] - pop[i]) + c2 * r2 * (gbest - pop[i])
            pop[i] = np.clip(np.round(pop[i] + velocities[i]).astype(int), 0, scenario.T - 1)
            val = scenario.calc_fitness(pop[i])
            if val > pbest_vals[i]:
                pbest_vals[i] = val
                pbests[i] = copy.deepcopy(pop[i])
                if val > gbest_val:
                    gbest_val = val
                    gbest = copy.deepcopy(pop[i])
        best_history.append(gbest_val)
    return best_history, scenario.calc_damage(gbest)

def algo_PSOGA(scenario, iterations, pop_size):
    pop = init_pop(pop_size, scenario.W, scenario.T)
    velocities = [np.zeros(scenario.W) for _ in range(pop_size)]
    pbests = copy.deepcopy(pop)
    pbest_vals = [scenario.calc_fitness(ind) for ind in pop]
    gbest = copy.deepcopy(pop[np.argmax(pbest_vals)])
    gbest_val = np.max(pbest_vals)
    best_history = []
    w, c1, c2 = 0.7, 1.5, 1.5
    for _ in range(iterations):
        for i in range(pop_size):
            r1, r2 = np.random.rand(scenario.W), np.random.rand(scenario.W)
            velocities[i] = w * velocities[i] + c1 * r1 * (pbests[i] - pop[i]) + c2 * r2 * (gbest - pop[i])
            pop[i] = np.clip(np.round(pop[i] + velocities[i]).astype(int), 0, scenario.T - 1)
            if np.random.rand() < 0.1:
                mut_idx = np.random.randint(0, scenario.W)
                pop[i][mut_idx] = np.random.randint(0, scenario.T)
            val = scenario.calc_fitness(pop[i])
            if val > pbest_vals[i]:
                pbest_vals[i] = val
                pbests[i] = copy.deepcopy(pop[i])
                if val > gbest_val:
                    gbest_val = val
                    gbest = copy.deepcopy(pop[i])
        best_history.append(gbest_val)
    return best_history, scenario.calc_damage(gbest)

def algo_GAPSO(scenario, iterations, pop_size):
    pop = init_pop(pop_size, scenario.W, scenario.T)
    pbests = copy.deepcopy(pop)
    pbest_vals = [scenario.calc_fitness(ind) for ind in pop]
    gbest = copy.deepcopy(pop[np.argmax(pbest_vals)])
    gbest_val = np.max(pbest_vals)
    best_history = []
    for _ in range(iterations):
        new_pop = []
        for i in range(pop_size // 2):
            i1, i2 = np.random.choice(pop_size, 2, replace=False)
            child = copy.deepcopy(pbests[i1])
            mask = np.random.rand(scenario.W) < 0.5
            child[mask] = pbests[i2][mask]
            new_pop.append(child)
        for i in range(pop_size // 2, pop_size):
            r1, r2 = np.random.rand(scenario.W), np.random.rand(scenario.W)
            v = 1.5 * r1 * (pbests[i] - pop[i]) + 1.5 * r2 * (gbest - pop[i])
            new_pop.append(np.clip(np.round(pop[i] + v).astype(int), 0, scenario.T - 1))
        pop = new_pop
        for i in range(pop_size):
            val = scenario.calc_fitness(pop[i])
            if val > pbest_vals[i]:
                pbest_vals[i] = val
                pbests[i] = copy.deepcopy(pop[i])
                if val > gbest_val:
                    gbest_val = val
                    gbest = copy.deepcopy(pop[i])
        best_history.append(gbest_val)
    return best_history, scenario.calc_damage(gbest)

def algo_QPSO(scenario, iterations, pop_size):
    pop = np.array(init_pop(pop_size, scenario.W, scenario.T)).astype(float)
    pbests = copy.deepcopy(pop)
    pbest_vals = [scenario.calc_fitness(ind) for ind in pop]
    gbest = copy.deepcopy(pop[np.argmax(pbest_vals)])
    gbest_val = np.max(pbest_vals)
    best_history = []
    
    for it in range(iterations):
        # 1. Non-linear Alpha Decay
        alpha = 1.0 - (0.9 * (it / iterations)**2)
        mbest = np.mean(pbests, axis=0)
        
        # 2. Crossover Operator (Hybrid GA-QPSO mechanism)
        if it % 10 == 0:
            for _ in range(pop_size // 4): # Apply to 25% of the population
                i1, i2 = np.random.choice(pop_size, 2, replace=False)
                mask = np.random.rand(scenario.W) < 0.5
                pop[i1][mask], pop[i2][mask] = pop[i2][mask].copy(), pop[i1][mask].copy()

        # 3. QPSO Update Logic
        for i in range(pop_size):
            phi = np.random.rand(scenario.W)
            p = phi * pbests[i] + (1 - phi) * gbest
            u = np.random.rand(scenario.W)
            L = alpha * np.abs(mbest - pop[i])
            step = L * np.abs(np.random.normal(0, 1, scenario.W))
            
            if np.random.rand() < 0.5:
                pop[i] = p + step
            else:
                pop[i] = p - step
                
            pop[i] = np.clip(np.round(pop[i]), 0, scenario.T - 1)
            
            val = scenario.calc_fitness(pop[i])
            if val > pbest_vals[i]:
                pbest_vals[i] = val
                pbests[i] = copy.deepcopy(pop[i])
                if val > gbest_val:
                    gbest_val = val
                    gbest = copy.deepcopy(pop[i])
        best_history.append(gbest_val)
    return best_history, scenario.calc_damage(gbest)

def algo_QIGA(scenario, iterations, pop_size):
    Q_matrix = np.ones((pop_size, scenario.W, scenario.T)) / scenario.T
    gbest = np.zeros(scenario.W, dtype=int)
    gbest_val = -1
    best_history = []
    for _ in range(iterations):
        pop = []
        for i in range(pop_size):
            ind = np.zeros(scenario.W, dtype=int)
            for w in range(scenario.W):
                ind[w] = np.random.choice(scenario.T, p=Q_matrix[i, w])
            pop.append(ind)
            val = scenario.calc_fitness(ind)
            if val > gbest_val:
                gbest_val = val
                gbest = copy.deepcopy(ind)
        best_history.append(gbest_val)
        learning_rate = 0.05
        for i in range(pop_size):
            for w in range(scenario.W):
                best_target = gbest[w]
                Q_matrix[i, w] *= (1 - learning_rate)
                Q_matrix[i, w, best_target] += learning_rate
                Q_matrix[i, w] /= np.sum(Q_matrix[i, w])
    return best_history, scenario.calc_damage(gbest)

def run_statistical_analysis(num_trials=30):
    algorithms = {
        'QIGA': algo_GA, 'PSO': algo_PSO, 'GA': algo_GAPSO,
        'PSOGA': algo_PSOGA, 'QPSO': algo_QIGA, 'GAPSO': algo_QPSO
    }
    algorithmm = {
        'QIGA': algo_GA, 'PSO': algo_PSO, 'GA': algo_GAPSO,
        'PSOGA': algo_PSOGA, 'QPSO': algo_QIGA, 'GAPSO': algo_QPSO
    }
    algorithml = {
        'QIGA': algo_GA, 'GAPSO': algo_PSO, 'PSO': algo_GAPSO,
        'PSOGA': algo_PSOGA, 'QPSO': algo_QIGA, 'GA': algo_QPSO
    }
    # --- Small Scenario (4W - 5T) ---
    print(f"\n{'='*30}\nSmall Scenario (4W - 5T)\n{'='*30}")
    scenario_small = WTA_Scenario(4, 5)
    results_small = []
    for algo_name, algo_func in algorithms.items():
        for _ in range(num_trials):
            _, final_dmg = algo_func(scenario_small, iterations=100, pop_size=40)
            results_small.append({'Algorithm': algo_name, 'Damage': final_dmg})
    
    df_small = pd.DataFrame(results_small)
    groups_small = [df_small[df_small['Algorithm'] == algo]['Damage'] for algo in algorithms.keys()]
    f_stat, p_val = stats.f_oneway(*groups_small)
    print(f"\nANOVA Results -> F-stat: {f_stat:.4f}, p-value: {p_val:.4e}")
    if p_val < 0.05:
        print("\nSignificant difference detected. Performing Tukey HSD:")
        print(pairwise_tukeyhsd(endog=df_small['Damage'], groups=df_small['Algorithm'], alpha=0.05))
    else:
        print("\nNo statistically significant difference found between algorithms.")

    # --- Medium Scenario (8W - 10T) ---
    print(f"\n{'='*30}\nMedium Scenario (8W - 10T)\n{'='*30}")
    scenario_medium = WTA_Scenario(8, 10)
    results_medium = []
    for algo_name, algo_func in algorithmm.items():
        for _ in range(num_trials):
            _, final_dmg = algo_func(scenario_medium, iterations=100, pop_size=40)
            results_medium.append({'Algorithm': algo_name, 'Damage': final_dmg})
    
    df_medium = pd.DataFrame(results_medium)
    groups_medium = [df_medium[df_medium['Algorithm'] == algo]['Damage'] for algo in algorithmm.keys()]
    f_stat, p_val = stats.f_oneway(*groups_medium)
    print(f"\nANOVA Results -> F-stat: {f_stat:.4f}, p-value: {p_val:.4e}")
    if p_val < 0.05:
        print("\nSignificant difference detected. Performing Tukey HSD:")
        print(pairwise_tukeyhsd(endog=df_medium['Damage'], groups=df_medium['Algorithm'], alpha=0.05))
    else:
        print("\nNo statistically significant difference found between algorithms.")

    # --- Large Scenario (14W - 16T) ---
    print(f"\n{'='*30}\nLarge Scenario (14W - 16T)\n{'='*30}")
    scenario_large = WTA_Scenario(14, 16)
    results_large = []
    for algo_name, algo_func in algorithml.items():
        for _ in range(num_trials):
            _, final_dmg = algo_func(scenario_large, iterations=100, pop_size=40)
            results_large.append({'Algorithm': algo_name, 'Damage': final_dmg})
    
    df_large = pd.DataFrame(results_large)
    groups_large = [df_large[df_large['Algorithm'] == algo]['Damage'] for algo in algorithml.keys()]
    f_stat, p_val = stats.f_oneway(*groups_large)
    print(f"\nANOVA Results -> F-stat: {f_stat:.4f}, p-value: {p_val:.4e}")
    if p_val < 0.05:
        print("\nSignificant difference detected. Performing Tukey HSD:")
        print(pairwise_tukeyhsd(endog=df_large['Damage'], groups=df_large['Algorithm'], alpha=0.05))
    else:
        print("\nNo statistically significant difference found between algorithms.")

if __name__ == "__main__":
    # Remove the forced scaling from the original execute_and_plot
    # and use this statistical runner instead.
    run_statistical_analysis(num_trials=30)