# 🚀 Weapon Target Assignment (WTA) Optimization using Classical, Hybrid & Quantum-Inspired Metaheuristic Algorithms

A comprehensive research implementation of the **Weapon Target Assignment (WTA)** optimization problem using **classical evolutionary algorithms, hybrid metaheuristics, and quantum-inspired optimization techniques**.

This repository presents a comparative study of six optimization algorithms on multiple benchmark WTA scenarios, demonstrating the effectiveness of **Quantum-Behaved Particle Swarm Optimization (QPSO)** for solving large-scale NP-hard optimization problems.

# 🎯 Introduction

The **Weapon Target Assignment (WTA)** problem is a classical **NP-hard combinatorial optimization problem** in Operations Research and Defense Systems.

Its objective is to determine the optimal allocation of available weapons to multiple targets while maximizing mission effectiveness and minimizing resource utilization under operational constraints.

Since exhaustive search becomes computationally infeasible as the number of weapons and targets increases, metaheuristic optimization algorithms provide practical near-optimal solutions within reasonable computational time.

This project implements, analyzes, and compares six different optimization techniques ranging from traditional evolutionary algorithms to advanced quantum-inspired approaches.

# 🧠 Implemented Algorithms

## 1. Genetic Algorithm (GA)

- Selection
- Crossover
- Mutation
- Evolutionary optimization


## 2. Particle Swarm Optimization (PSO)

- Population-based swarm intelligence
- Personal Best (pBest)
- Global Best (gBest)
- Velocity-position update mechanism


## 3. Hybrid GA-PSO

Combines:

- GA exploration
- PSO exploitation

to improve convergence while maintaining diversity.


## 4. Hybrid PSO-GA

A two-stage optimization approach:

1. Global search using PSO
2. Local refinement using GA


## 5. Quantum-Inspired Genetic Algorithm (QIGA)

Implements quantum-inspired evolutionary concepts:

- Qubit chromosome representation
- Probability amplitudes
- Quantum rotation gates
- Enhanced population diversity


## 6. Quantum-Behaved Particle Swarm Optimization (QPSO)

A quantum variant of PSO based on quantum mechanics principles.

Key features include:

- Delta potential well model
- Mean Best (mbest)
- Local attractor computation
- Quantum position update
- Contraction-Expansion coefficient (β)

QPSO removes the velocity term used in classical PSO and significantly improves global search capability while reducing the likelihood of premature convergence.


# ✨ Features

- Complete implementation of six optimization algorithms
- Modular Python implementation
- Classical and quantum-inspired optimization techniques
- Comparative performance analysis
- Small, medium and large benchmark scenarios
- Convergence plots
- Fitness comparison
- Statistical validation
- Easy-to-modify parameters
- Research-oriented implementation


# 📊 Scenario Configuration

Three benchmark WTA scenarios are considered.

| Scenario | Weapons | Targets | Search Dimension |
|----------|:-------:|:-------:|:----------------:|
| Small | 4 | 5 | 20 |
| Medium | 8 | 10 | 80 |
| Large | 14 | 16 | 224 |

Increasing the number of weapons and targets significantly enlarges the search space, making the optimization problem progressively more challenging.

These benchmark scenarios evaluate both optimization quality and scalability.


# 📈 Performance Evaluation Metrics

Each algorithm is evaluated using:

- Best Fitness
- Average Fitness
- Worst Fitness
- Median Fitness
- Standard Deviation
- Computational Time
- Convergence Speed
- Target Neutralization Efficiency
- Exploration–Exploitation Balance
- Solution Stability


# 📉 Statistical Validation

To verify that the observed improvements are statistically significant, the project includes:

- Descriptive Statistics
- Mean Comparison
- Variance Comparison
- One-Way ANOVA
- Tukey HSD Post-Hoc Analysis
- Confidence Interval Analysis

These statistical tests provide quantitative evidence of the superiority of quantum-inspired optimization algorithms.


# 🏆 Results Summary

Experimental results demonstrate consistent improvements in optimization performance with quantum-inspired algorithms.

Typical ranking obtained across benchmark scenarios:

GA
    ↓
PSO
    ↓
Hybrid GA-PSO
    ↓
Hybrid PSO-GA
    ↓
QIGA
    ↓
QPSO


Key observations:

- Faster convergence
- Better global search capability
- Higher fitness values
- Reduced premature convergence
- Improved scalability
- Better solution stability

Among all implemented algorithms, **Quantum-Behaved Particle Swarm Optimization (QPSO)** consistently achieved the best overall performance.


# 📚 Applications

- Weapon Target Assignment
- Military Decision Support Systems
- Defense Resource Allocation
- Mission Planning
- Operations Research
- Swarm Intelligence
- Evolutionary Computing
- Quantum-Inspired Optimization
- Combinatorial Optimization
- Artificial Intelligence Research