"""
Main module for running and comparing different TSP algorithms.

This module provides a unified interface to run and compare the performance
of different algorithms for solving the Traveling Salesman Problem.

Author: Your Name
Date: 2024
"""

import time
import pandas as pd
from typing import List, Dict, Tuple
from genetic_algorithm import GeneticAlgorithm
from simulated_annealing import SimulatedAnnealing
from tabu_search import TabuSearch
from particle_swarm import ParticleSwarmOptimization

def load_cities(file_path: str) -> List[Dict]:
    """
    Load city data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing city data

    Returns:
        List[Dict]: List of cities with their coordinates
    """
    df = pd.read_csv(file_path)
    return df.to_dict('records')

def run_algorithm(algorithm_name: str, cities: List[Dict]) -> Tuple[List[int], float, float]:
    """
    Run a specific algorithm and measure its performance.

    Args:
        algorithm_name (str): Name of the algorithm to run
        cities (List[Dict]): List of cities with their coordinates

    Returns:
        Tuple[List[int], float, float]: (best_route, distance, execution_time)
    """
    # Initialize algorithm
    if algorithm_name == "Genetic Algorithm":
        solver = GeneticAlgorithm(cities)
    elif algorithm_name == "Simulated Annealing":
        solver = SimulatedAnnealing(cities)
    elif algorithm_name == "Tabu Search":
        solver = TabuSearch(cities)
    elif algorithm_name == "Particle Swarm Optimization":
        solver = ParticleSwarmOptimization(cities)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    # Run algorithm and measure time
    start_time = time.time()
    best_route, distance = solver.solve()
    execution_time = time.time() - start_time

    return best_route, distance, execution_time

def compare_algorithms(cities: List[Dict]) -> pd.DataFrame:
    """
    Compare the performance of all algorithms.

    Args:
        cities (List[Dict]): List of cities with their coordinates

    Returns:
        pd.DataFrame: Comparison results
    """
    algorithms = [
        "Genetic Algorithm",
        "Simulated Annealing",
        "Tabu Search",
        "Particle Swarm Optimization"
    ]

    results = []
    for algorithm in algorithms:
        print(f"Running {algorithm}...")
        route, distance, time_taken = run_algorithm(algorithm, cities)
        results.append({
            "Algorithm": algorithm,
            "Best Distance (km)": distance,
            "Execution Time (s)": time_taken
        })

    # Create DataFrame and calculate scores
    df = pd.DataFrame(results)
    
    # Calculate scores (0-100 scale)
    min_distance = df["Best Distance (km)"].min()
    max_distance = df["Best Distance (km)"].max()
    min_time = df["Execution Time (s)"].min()
    max_time = df["Execution Time (s)"].max()
    
    df["Distance Score"] = 100 * (1 - (df["Best Distance (km)"] - min_distance) / (max_distance - min_distance))
    df["Speed Score"] = 100 * (1 - (df["Execution Time (s)"] - min_time) / (max_time - min_time))
    df["Overall Score"] = (df["Distance Score"] + df["Speed Score"]) / 2
    
    return df

def main():
    """Main function to run the comparison."""
    # Load city data
    cities = load_cities("data.csv")
    
    # Compare algorithms
    results = compare_algorithms(cities)
    
    # Save results
    results.to_csv("algorithm_comparison.csv", index=False)
    print("\nResults saved to algorithm_comparison.csv")
    
    # Print results
    print("\nAlgorithm Comparison Results:")
    print(results.to_string(index=False))

if __name__ == "__main__":
    main() 