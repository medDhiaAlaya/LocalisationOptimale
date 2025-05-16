"""
Simulated Annealing Implementation for Traveling Salesman Problem (TSP)

This module implements a Simulated Annealing algorithm to solve the TSP, which finds
the shortest possible route that visits each city exactly once and returns to the origin city.

Time Complexity: O(i * n²)
- i: number of iterations
- n: number of cities

Author: Your Name
Date: 2024
"""

import numpy as np
import random
from typing import List, Tuple, Dict

class SimulatedAnnealing:
    def __init__(self, cities: List[Dict], initial_temp: float = 100.0,
                 cooling_rate: float = 0.95, iterations: int = 1000):
        """
        Initialize the Simulated Annealing solver.

        Args:
            cities (List[Dict]): List of cities with their coordinates
            initial_temp (float): Initial temperature for the annealing process
            cooling_rate (float): Rate at which temperature decreases
            iterations (int): Number of iterations at each temperature
        """
        self.cities = cities
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iterations = iterations
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

    def get_neighbor(self, route: List[int]) -> List[int]:
        """
        Generate a neighboring solution by swapping two random cities.

        Args:
            route (List[int]): Current route

        Returns:
            List[int]: New route with two cities swapped
        """
        new_route = route.copy()
        i, j = random.sample(range(len(route)), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]
        return new_route

    def acceptance_probability(self, current_distance: float, 
                             new_distance: float, temperature: float) -> float:
        """
        Calculate the probability of accepting a worse solution.

        Args:
            current_distance (float): Distance of current solution
            new_distance (float): Distance of new solution
            temperature (float): Current temperature

        Returns:
            float: Probability of accepting the new solution
        """
        if new_distance < current_distance:
            return 1.0
        return np.exp((current_distance - new_distance) / temperature)

    def solve(self) -> Tuple[List[int], float]:
        """
        Solve the TSP using Simulated Annealing.

        Returns:
            Tuple[List[int], float]: Best route found and its distance
        """
        # Initialize with a random route
        current_route = list(range(len(self.cities)))
        random.shuffle(current_route)
        current_distance = self.calculate_distance(current_route)
        
        self.best_solution = current_route.copy()
        self.best_distance = current_distance
        
        temperature = self.initial_temp
        
        while temperature > 1:
            for _ in range(self.iterations):
                # Generate new solution
                new_route = self.get_neighbor(current_route)
                new_distance = self.calculate_distance(new_route)
                
                # Decide whether to accept the new solution
                if self.acceptance_probability(current_distance, new_distance, temperature) > random.random():
                    current_route = new_route
                    current_distance = new_distance
                    
                    # Update best solution if necessary
                    if current_distance < self.best_distance:
                        self.best_solution = current_route.copy()
                        self.best_distance = current_distance
            
            # Cool down
            temperature *= self.cooling_rate
        
        return self.best_solution, self.best_distance 