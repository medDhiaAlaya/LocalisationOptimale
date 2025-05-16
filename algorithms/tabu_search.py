"""
Tabu Search Implementation for Traveling Salesman Problem (TSP)

This module implements a Tabu Search algorithm to solve the TSP, which finds
the shortest possible route that visits each city exactly once and returns to the origin city.

Time Complexity: O(i * n²)
- i: number of iterations
- n: number of cities

Author: Your Name
Date: 2024
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Set
from collections import deque

class TabuSearch:
    def __init__(self, cities: List[Dict], tabu_size: int = 10,
                 max_iterations: int = 1000, aspiration_criteria: bool = True):
        """
        Initialize the Tabu Search solver.

        Args:
            cities (List[Dict]): List of cities with their coordinates
            tabu_size (int): Size of the tabu list
            max_iterations (int): Maximum number of iterations
            aspiration_criteria (bool): Whether to use aspiration criteria
        """
        self.cities = cities
        self.tabu_size = tabu_size
        self.max_iterations = max_iterations
        self.aspiration_criteria = aspiration_criteria
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

    def get_neighbors(self, route: List[int]) -> List[Tuple[List[int], Tuple[int, int]]]:
        """
        Generate all possible neighboring solutions by swapping two cities.

        Args:
            route (List[int]): Current route

        Returns:
            List[Tuple[List[int], Tuple[int, int]]]: List of (neighbor, move) pairs
        """
        neighbors = []
        for i in range(len(route)):
            for j in range(i + 1, len(route)):
                new_route = route.copy()
                new_route[i], new_route[j] = new_route[j], new_route[i]
                neighbors.append((new_route, (i, j)))
        return neighbors

    def solve(self) -> Tuple[List[int], float]:
        """
        Solve the TSP using Tabu Search.

        Returns:
            Tuple[List[int], float]: Best route found and its distance
        """
        # Initialize with a random route
        current_route = list(range(len(self.cities)))
        random.shuffle(current_route)
        current_distance = self.calculate_distance(current_route)
        
        self.best_solution = current_route.copy()
        self.best_distance = current_distance
        
        # Initialize tabu list
        tabu_list = deque(maxlen=self.tabu_size)
        
        for _ in range(self.max_iterations):
            # Get all neighbors
            neighbors = self.get_neighbors(current_route)
            best_neighbor = None
            best_neighbor_distance = float('inf')
            best_move = None
            
            # Find the best non-tabu neighbor
            for neighbor, move in neighbors:
                neighbor_distance = self.calculate_distance(neighbor)
                
                # Check if move is tabu
                if move in tabu_list:
                    # Apply aspiration criteria if enabled
                    if self.aspiration_criteria and neighbor_distance < self.best_distance:
                        pass  # Override tabu status
                    else:
                        continue
                
                if neighbor_distance < best_neighbor_distance:
                    best_neighbor = neighbor
                    best_neighbor_distance = neighbor_distance
                    best_move = move
            
            # If no valid neighbor found, break
            if best_neighbor is None:
                break
            
            # Update current solution
            current_route = best_neighbor
            current_distance = best_neighbor_distance
            
            # Update tabu list
            tabu_list.append(best_move)
            
            # Update best solution if necessary
            if current_distance < self.best_distance:
                self.best_solution = current_route.copy()
                self.best_distance = current_distance
        
        return self.best_solution, self.best_distance 