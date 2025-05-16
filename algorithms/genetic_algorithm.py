"""
Genetic Algorithm Implementation for Traveling Salesman Problem (TSP)

This module implements a Genetic Algorithm to solve the TSP, which finds the shortest
possible route that visits each city exactly once and returns to the origin city.

Time Complexity: O(g * p * n²)
- g: number of generations
- p: population size
- n: number of cities

Author: Your Name
Date: 2024
"""

import numpy as np
import random
from typing import List, Tuple, Dict

class GeneticAlgorithm:
    def __init__(self, cities: List[Dict], population_size: int = 50, 
                 generations: int = 100, mutation_rate: float = 0.01):
        """
        Initialize the Genetic Algorithm solver.

        Args:
            cities (List[Dict]): List of cities with their coordinates
            population_size (int): Number of solutions in each generation
            generations (int): Number of generations to evolve
            mutation_rate (float): Probability of mutation for each gene
        """
        self.cities = cities
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.best_solution = None
        self.best_distance = float('inf')
        
    def calculate_distance(self, route: List[int]) -> float:
        """
        Calculate the total distance of a route.

        Args:
            route (List[int]): List of city indices representing the route

        Returns:
            float: Total distance of the route
        """
        total_distance = 0
        for i in range(len(route)):
            city1 = self.cities[route[i]]
            city2 = self.cities[route[(i + 1) % len(route)]]
            total_distance += np.sqrt(
                (city1['x'] - city2['x'])**2 + 
                (city1['y'] - city2['y'])**2
            )
        return total_distance

    def create_initial_population(self) -> List[List[int]]:
        """
        Create the initial population of routes.

        Returns:
            List[List[int]]: List of random routes
        """
        population = []
        for _ in range(self.population_size):
            route = list(range(len(self.cities)))
            random.shuffle(route)
            population.append(route)
        return population

    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Perform ordered crossover between two parents to create two children.

        Args:
            parent1 (List[int]): First parent route
            parent2 (List[int]): Second parent route

        Returns:
            Tuple[List[int], List[int]]: Two child routes
        """
        size = len(parent1)
        # Choose random crossover points
        point1, point2 = sorted(random.sample(range(size), 2))
        
        # Create children
        child1 = [-1] * size
        child2 = [-1] * size
        
        # Copy the segment between crossover points
        child1[point1:point2] = parent1[point1:point2]
        child2[point1:point2] = parent2[point1:point2]
        
        # Fill the remaining positions
        remaining1 = [x for x in parent2 if x not in child1[point1:point2]]
        remaining2 = [x for x in parent1 if x not in child2[point1:point2]]
        
        j1 = j2 = 0
        for i in range(size):
            if child1[i] == -1:
                child1[i] = remaining1[j1]
                j1 += 1
            if child2[i] == -1:
                child2[i] = remaining2[j2]
                j2 += 1
                
        return child1, child2

    def mutate(self, route: List[int]) -> List[int]:
        """
        Perform mutation on a route by swapping two random cities.

        Args:
            route (List[int]): Route to mutate

        Returns:
            List[int]: Mutated route
        """
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route

    def select_parents(self, population: List[List[int]], 
                      fitness_scores: List[float]) -> Tuple[List[int], List[int]]:
        """
        Select two parents using tournament selection.

        Args:
            population (List[List[int]]): Current population
            fitness_scores (List[float]): Fitness scores for each route

        Returns:
            Tuple[List[int], List[int]]: Two selected parent routes
        """
        tournament_size = 5
        parents = []
        for _ in range(2):
            tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
            winner = min(tournament, key=lambda x: x[1])[0]
            parents.append(winner)
        return tuple(parents)

    def solve(self) -> Tuple[List[int], float]:
        """
        Solve the TSP using the Genetic Algorithm.

        Returns:
            Tuple[List[int], float]: Best route found and its distance
        """
        # Initialize population
        population = self.create_initial_population()
        
        for generation in range(self.generations):
            # Calculate fitness for each route
            fitness_scores = [self.calculate_distance(route) for route in population]
            
            # Update best solution
            min_idx = np.argmin(fitness_scores)
            if fitness_scores[min_idx] < self.best_distance:
                self.best_distance = fitness_scores[min_idx]
                self.best_solution = population[min_idx].copy()
            
            # Create new population
            new_population = []
            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = self.select_parents(population, fitness_scores)
                
                # Create children
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutate children
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Replace old population with new one
            population = new_population[:self.population_size]
        
        return self.best_solution, self.best_distance 