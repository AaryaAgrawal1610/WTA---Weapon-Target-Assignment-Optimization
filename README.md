# WTA — Weapon Target Assignment Optimization

An advanced Python implementation of the **Weapon Target Assignment (WTA)** optimization problem using classical, hybrid, and quantum-inspired metaheuristic algorithms. This project provides a comparative analysis of different optimization techniques to maximize target neutralization efficiency while minimizing resource utilization.

## Overview

The **Weapon Target Assignment (WTA)** problem is a well-known NP-hard optimization problem in military operations research. It involves assigning available weapons to multiple targets in a way that maximizes mission effectiveness under operational constraints.

This repository implements and compares:

- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)
- Hybrid GA-PSO
- Hybrid PSO-GA
- Quantum-Inspired Genetic Algorithm (QIGA)
- Quantum-Behaved Particle Swarm Optimization (QPSO)

The project also includes scenario generation and performance comparison among all implemented algorithms.

## Algorithms Implemented

### Genetic Algorithm (GA)

A population-based evolutionary algorithm that applies selection, crossover, and mutation to evolve high-quality weapon-target assignments.

### Particle Swarm Optimization (PSO)

A swarm intelligence algorithm where particles update their positions using personal and global best solutions.

### Hybrid GA-PSO

Combines the exploration capability of Genetic Algorithms with the exploitation strength of Particle Swarm Optimization.

### Hybrid PSO-GA

Performs optimization using PSO followed by GA refinement to improve solution quality.

### Quantum-Inspired Genetic Algorithm (QIGA)

Utilizes quantum-inspired qubits and probability amplitudes to maintain solution diversity and enhance global search performance.

### Quantum-Behaved Particle Swarm Optimization (QPSO)

A quantum variant of PSO where particles move according to quantum mechanics principles, improving convergence and avoiding local optima.

## Technologies Used

- Python3
- NumPy
- Matplotlib
- Random
- Math

## Performance Comparison

The repository compares the algorithms based on:

- Best Fitness
- Average Fitness
- Convergence Speed
- Computational Time
- Solution Stability
- Exploration vs Exploitation
- Target Neutralization Efficiency

## Scenario Configuration

To evaluate the scalability and robustness of the implemented optimization algorithms, the Weapon Target Assignment (WTA) problem is tested on three benchmark scenarios of increasing complexity.

| Scenario | Weapons | Targets | Problem Dimension |
|----------|:-------:|:-------:|:----------------:|
| Small Scale | 4 | 5 | 20 |
| Medium Scale | 8 | 10 | 80 |
| Large Scale | 14 | 16 | 224 |

The increase in the number of weapons and targets significantly expands the search space, making the optimization problem progressively more challenging. These benchmark scenarios enable a comprehensive comparison of algorithm performance under varying computational complexities.

The implemented algorithms are evaluated across all three scenarios using the following performance metrics:

- Best Fitness
- Average Fitness
- Computational Time
- Convergence Speed
- Solution Stability
- Target Neutralization Efficiency
- Exploration and Exploitation Balance

The small-scale scenario validates algorithm correctness, the medium-scale scenario assesses optimization capability under moderate complexity, and the large-scale scenario demonstrates scalability and robustness for solving high-dimensional Weapon Target Assignment problems.

## Expected Outcome

The comparison demonstrates the effectiveness of quantum-inspired optimization techniques over traditional metaheuristic approaches for solving the Weapon Target Assignment problem.

Typical performance ranking:

GA < PSO < Hybrid GA-PSO < Hybrid PSO-GA < QIGA < QPSO

## Applications

- Defense Resource Allocation
- Weapon Target Assignment
- Mission Planning
- Military Decision Support Systems
- Swarm Intelligence Research
- Evolutionary Computing
- Quantum-Inspired Optimization
- Operations Research
