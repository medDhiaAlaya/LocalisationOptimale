"""
Particle Swarm Optimization (PSO) Implementation for Traveling Salesman Problem (TSP)

This module implements a PSO algorithm to solve the TSP, which finds the shortest
possible route that visits each city exactly once and returns to the origin city.

Time Complexity: O(i * p * n²)
- i: number of iterations
- p: number of particles
- n: number of cities

Author: Your Name
Date: 2024
"""

import numpy as np
import random
from typing import List, Tuple, Dict

class ParticleSwarmOptimization:
    def __init__(self, cities: List[Dict], num_particles: int = 30,
                 max_iterations: int = 100, w: float = 0.7,
                 c1: float = 1.5, c2: float = 1.5):
        """
        Initialize the PSO solver.

        Args:
            cities (List[Dict]): List of cities with their coordinates
            num_particles (int): Number of particles in the swarm
            max_iterations (int): Maximum number of iterations
            w (float): Inertia weight
            c1 (float): Cognitive parameter
            c2 (float): Social parameter
        """
        self.cities = cities
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
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

    def initialize_particle(self) -> Tuple[List[int], List[float], List[float]]:
        """
        Initialize a particle with a random route and velocity.

        Returns:
            Tuple[List[int], List[float], List[float]]: (route, velocity, best_route)
        """
        route = list(range(len(self.cities)))
        random.shuffle(route)
        velocity = [random.uniform(-1, 1) for _ in range(len(self.cities))]
        return route, velocity, route.copy()

    def update_velocity(self, velocity: List[float], current: List[int],
                       pbest: List[int], gbest: List[int]) -> List[float]:
        """
        Update particle velocity using PSO equations.

        Args:
            velocity (List[float]): Current velocity
            current (List[int]): Current position
            pbest (List[int]): Personal best position
            gbest (List[int]): Global best position

        Returns:
            List[float]: Updated velocity
        """
        new_velocity = []
        for i in range(len(velocity)):
            r1, r2 = random.random(), random.random()
            cognitive = self.c1 * r1 * (pbest[i] - current[i])
            social = self.c2 * r2 * (gbest[i] - current[i])
            new_velocity.append(self.w * velocity[i] + cognitive + social)
        return new_velocity

    def update_position(self, position: List[int], velocity: List[float]) -> List[int]:
        """
        Update particle position based on velocity.

        Args:
            position (List[int]): Current position
            velocity (List[float]): Current velocity

        Returns:
            List[int]: Updated position
        """
        # Convert velocity to swap probabilities
        probs = [1 / (1 + np.exp(-v)) for v in velocity]
        
        # Apply swaps based on probabilities
        new_position = position.copy()
        for i in range(len(position)):
            if random.random() < probs[i]:
                j = random.randrange(len(position))
                new_position[i], new_position[j] = new_position[j], new_position[i]
        
        return new_position

    def solve(self) -> Tuple[List[int], float]:
        """
        Solve the TSP using Particle Swarm Optimization.

        Returns:
            Tuple[List[int], float]: Best route found and its distance
        """
        # Initialize particles
        particles = []
        pbest_routes = []
        pbest_distances = []
        
        for _ in range(self.num_particles):
            route, velocity, best_route = self.initialize_particle()
            distance = self.calculate_distance(route)
            particles.append((route, velocity))
            pbest_routes.append(best_route)
            pbest_distances.append(distance)
            
            if distance < self.best_distance:
                self.best_solution = route.copy()
                self.best_distance = distance
        
        # Main PSO loop
        for _ in range(self.max_iterations):
            for i in range(self.num_particles):
                route, velocity = particles[i]
                
                # Update velocity
                new_velocity = self.update_velocity(
                    velocity, route, pbest_routes[i], self.best_solution
                )
                
                # Update position
                new_route = self.update_position(route, new_velocity)
                new_distance = self.calculate_distance(new_route)
                
                # Update personal best
                if new_distance < pbest_distances[i]:
                    pbest_routes[i] = new_route.copy()
                    pbest_distances[i] = new_distance
                    
                    # Update global best
                    if new_distance < self.best_distance:
                        self.best_solution = new_route.copy()
                        self.best_distance = new_distance
                
                # Update particle
                particles[i] = (new_route, new_velocity)
        
        return self.best_solution, self.best_distance 