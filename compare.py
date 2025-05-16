import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from tqdm import tqdm
import seaborn as sns

# Set the random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Load the data from CSV file
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Convert lat and lng to float
    df['lat'] = df['lat'].astype(float)
    df['lng'] = df['lng'].astype(float)
    # We'll use only cities with population data
    df = df[df['population'].notna()]
    df['population'] = df['population'].astype(float)
    return df

# Our optimization problem: Find the best route visiting the 20 most populated cities
# We'll use the Traveling Salesman Problem (TSP) as our benchmark
class OptimizationProblem:
    def __init__(self, df, num_cities=20):
        # Select the top cities by population
        self.cities = df.nlargest(num_cities, 'population')
        self.num_cities = len(self.cities)
        # Create a distance matrix
        self.distance_matrix = self._create_distance_matrix()
        # The optimal solution is unknown, so we'll track the best found
        self.best_known_solution = None
        self.best_known_distance = float('inf')
    
    def _create_distance_matrix(self):
        """Create a matrix of distances between all cities."""
        n = self.num_cities
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Calculate Haversine distance between cities
                    lat1, lng1 = self.cities.iloc[i]['lat'], self.cities.iloc[i]['lng']
                    lat2, lng2 = self.cities.iloc[j]['lat'], self.cities.iloc[j]['lng']
                    matrix[i, j] = self._haversine_distance(lat1, lng1, lat2, lng2)
        return matrix
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the Haversine distance between two points on Earth."""
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        return c * r
    
    def evaluate_solution(self, solution):
        """Calculate the total distance of a tour."""
        total_distance = 0
        for i in range(len(solution)):
            from_city = solution[i]
            to_city = solution[(i + 1) % len(solution)]
            total_distance += self.distance_matrix[from_city, to_city]
        return total_distance
    
    def random_solution(self):
        """Generate a random solution."""
        solution = list(range(self.num_cities))
        random.shuffle(solution)
        return solution
    
    def get_city_names(self, solution):
        """Get the names of cities in a solution."""
        return [self.cities.iloc[i]['city'] for i in solution]

# 1. Genetic Algorithm (GA)
class GeneticAlgorithm:
    def __init__(self, problem, pop_size=100, elite_size=20, mutation_rate=0.01, generations=100):
        self.problem = problem
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.history = []
    
    def initial_population(self):
        """Create an initial population of random solutions."""
        population = []
        for _ in range(self.pop_size):
            population.append(self.problem.random_solution())
        return population
    
    def rank_solutions(self, population):
        """Rank solutions by fitness (lower distance is better)."""
        fitness_results = {}
        for i in range(len(population)):
            fitness_results[i] = self.problem.evaluate_solution(population[i])
        return sorted(fitness_results.items(), key=lambda x: x[1])
    
    def selection(self, ranked_population):
        """Select parents for breeding."""
        selection_results = []
        # Add elite individuals
        for i in range(self.elite_size):
            selection_results.append(ranked_population[i][0])
        
        # Roulette wheel selection for the rest
        df = pd.DataFrame(np.array(ranked_population), columns=["Index", "Fitness"])
        df['Fitness'] = 1 / df['Fitness']  # Invert fitness since lower distance is better
        df['cum_sum'] = df['Fitness'].cumsum()
        df['cum_perc'] = 100 * df['cum_sum'] / df['Fitness'].sum()
        
        for _ in range(0, self.pop_size - self.elite_size):
            pick = 100 * random.random()
            for i in range(len(ranked_population)):
                if pick <= df.iloc[i]['cum_perc']:
                    selection_results.append(ranked_population[i][0])
                    break
        
        return selection_results
    
    def mating_pool(self, population, selection_results):
        """Create the mating pool."""
        pool = []
        for i in selection_results:
            pool.append(population[i])
        return pool
    
    def breed(self, parent1, parent2):
        """Create a child using ordered crossover."""
        child = [-1] * len(parent1)
        
        # Get subset of parent1
        start_gene, end_gene = sorted(random.sample(range(len(parent1)), 2))
        for i in range(start_gene, end_gene + 1):
            child[i] = parent1[i]
        
        # Fill the remaining positions with values from parent2
        pointer = 0
        for i in range(len(parent2)):
            if pointer == len(child):
                break
            if child[pointer] == -1:
                if parent2[i] not in child:
                    child[pointer] = parent2[i]
                    pointer += 1
            else:
                pointer += 1
        
        # Fill any remaining positions
        for i in range(len(child)):
            if child[i] == -1:
                for j in range(len(parent2)):
                    if parent2[j] not in child:
                        child[i] = parent2[j]
                        break
        
        return child
    
    def breed_population(self, mating_pool):
        """Create a new generation through breeding."""
        children = []
        # Keep elite individuals
        for i in range(self.elite_size):
            children.append(mating_pool[i])
        
        # Breed the rest
        for i in range(self.pop_size - self.elite_size):
            parent1 = random.choice(mating_pool)
            parent2 = random.choice(mating_pool)
            children.append(self.breed(parent1, parent2))
        
        return children
    
    def mutate(self, individual):
        """Apply swap mutation to an individual."""
        for swapped in range(len(individual)):
            if random.random() < self.mutation_rate:
                swap_with = int(random.random() * len(individual))
                individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]
        return individual
    
    def mutate_population(self, population):
        """Apply mutation to the entire population."""
        mutated_pop = []
        for ind in population:
            mutated = self.mutate(ind.copy())
            mutated_pop.append(mutated)
        return mutated_pop
    
    def next_generation(self, current_gen):
        """Create the next generation."""
        ranked_pop = self.rank_solutions(current_gen)
        selection_results = self.selection(ranked_pop)
        mating_pool = self.mating_pool(current_gen, selection_results)
        children = self.breed_population(mating_pool)
        next_gen = self.mutate_population(children)
        return next_gen
    
    def solve(self):
        """Run the genetic algorithm for the specified number of generations."""
        start_time = time.time()
        population = self.initial_population()
        best_distance = float('inf')
        best_solution = None
        
        for i in tqdm(range(self.generations), desc="GA Progress"):
            population = self.next_generation(population)
            ranked_pop = self.rank_solutions(population)
            current_best_idx = ranked_pop[0][0]
            current_best_distance = ranked_pop[0][1]
            
            if current_best_distance < best_distance:
                best_distance = current_best_distance
                best_solution = population[current_best_idx]
            
            self.history.append(best_distance)
        
        end_time = time.time()
        
        return {
            'algorithm': 'Genetic Algorithm',
            'solution': best_solution,
            'distance': best_distance,
            'time': end_time - start_time,
            'history': self.history
        }

# 2. Simulated Annealing (SA)
class SimulatedAnnealing:
    def __init__(self, problem, initial_temp=1000, cooling_rate=0.003, iterations=1000):
        self.problem = problem
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iterations = iterations
        self.history = []
    
    def solve(self):
        """Run the simulated annealing algorithm."""
        start_time = time.time()
        
        # Initialize with a random solution
        current_solution = self.problem.random_solution()
        current_distance = self.problem.evaluate_solution(current_solution)
        best_solution = current_solution.copy()
        best_distance = current_distance
        
        temperature = self.initial_temp
        
        for i in tqdm(range(self.iterations), desc="SA Progress"):
            # Create a new candidate solution by swapping two cities
            new_solution = current_solution.copy()
            pos1, pos2 = random.sample(range(len(new_solution)), 2)
            new_solution[pos1], new_solution[pos2] = new_solution[pos2], new_solution[pos1]
            
            # Calculate the new distance
            new_distance = self.problem.evaluate_solution(new_solution)
            
            # Decide whether to accept the new solution
            if new_distance < current_distance:
                # Accept better solution
                current_solution = new_solution
                current_distance = new_distance
                
                # Update best if new solution is better
                if new_distance < best_distance:
                    best_solution = new_solution.copy()
                    best_distance = new_distance
            else:
                # Accept worse solution with a probability based on temperature
                delta = new_distance - current_distance
                probability = np.exp(-delta / temperature)
                if random.random() < probability:
                    current_solution = new_solution
                    current_distance = new_distance
            
            # Cool down the temperature
            temperature *= (1 - self.cooling_rate)
            
            # Record the best distance so far
            self.history.append(best_distance)
        
        end_time = time.time()
        
        return {
            'algorithm': 'Simulated Annealing',
            'solution': best_solution,
            'distance': best_distance,
            'time': end_time - start_time,
            'history': self.history
        }

# 3. Tabu Search
class TabuSearch:
    def __init__(self, problem, tabu_size=20, iterations=1000):
        self.problem = problem
        self.tabu_size = tabu_size
        self.iterations = iterations
        self.history = []
    
    def _get_neighbors(self, solution):
        """Generate all neighbors by swapping pairs of cities."""
        neighbors = []
        for i in range(len(solution)):
            for j in range(i+1, len(solution)):
                neighbor = solution.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append((neighbor, (i, j)))
        return neighbors
    
    def solve(self):
        """Run the tabu search algorithm."""
        start_time = time.time()
        
        # Initialize with a random solution
        current_solution = self.problem.random_solution()
        current_distance = self.problem.evaluate_solution(current_solution)
        best_solution = current_solution.copy()
        best_distance = current_distance
        
        # Initialize tabu list
        tabu_list = []
        
        for i in tqdm(range(self.iterations), desc="Tabu Search Progress"):
            # Get all neighbors of the current solution
            neighbors = self._get_neighbors(current_solution)
            
            # Find the best non-tabu neighbor
            best_neighbor = None
            best_neighbor_distance = float('inf')
            best_move = None
            
            for neighbor, move in neighbors:
                # Skip if the move is in the tabu list
                if move in tabu_list or (move[1], move[0]) in tabu_list:
                    continue
                
                # Evaluate the neighbor
                distance = self.problem.evaluate_solution(neighbor)
                
                # Update best neighbor
                if distance < best_neighbor_distance:
                    best_neighbor = neighbor
                    best_neighbor_distance = distance
                    best_move = move
            
            # If no non-tabu neighbor found, use the best neighbor regardless of tabu
            if best_neighbor is None:
                for neighbor, move in neighbors:
                    distance = self.problem.evaluate_solution(neighbor)
                    if distance < best_neighbor_distance:
                        best_neighbor = neighbor
                        best_neighbor_distance = distance
                        best_move = move
            
            # Update current solution
            current_solution = best_neighbor
            current_distance = best_neighbor_distance
            
            # Update best solution if improved
            if current_distance < best_distance:
                best_solution = current_solution.copy()
                best_distance = current_distance
            
            # Add the move to the tabu list
            tabu_list.append(best_move)
            if len(tabu_list) > self.tabu_size:
                tabu_list.pop(0)
            
            # Record the best distance so far
            self.history.append(best_distance)
        
        end_time = time.time()
        
        return {
            'algorithm': 'Tabu Search',
            'solution': best_solution,
            'distance': best_distance,
            'time': end_time - start_time,
            'history': self.history
        }

# 4. Particle Swarm Optimization (PSO)
class ParticleSwarmOptimization:
    def __init__(self, problem, num_particles=30, w=0.5, c1=1, c2=2, iterations=100):
        self.problem = problem
        self.num_particles = num_particles
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter
        self.iterations = iterations
        self.history = []
    
    def _create_particles(self):
        """Initialize particles with random positions and velocities."""
        particles = []
        for _ in range(self.num_particles):
            # Position is a valid tour
            position = self.problem.random_solution()
            # Velocity is a list of swap operations
            velocity = []
            for _ in range(random.randint(0, 5)):
                i, j = random.sample(range(len(position)), 2)
                velocity.append((i, j))
            
            # Personal best
            pbest_pos = position.copy()
            pbest_val = self.problem.evaluate_solution(position)
            
            particles.append({
                'position': position,
                'velocity': velocity,
                'pbest_pos': pbest_pos,
                'pbest_val': pbest_val
            })
        
        return particles
    
    def _apply_velocity(self, position, velocity):
        """Apply swap operations to a position."""
        new_position = position.copy()
        for i, j in velocity:
            new_position[i], new_position[j] = new_position[j], new_position[i]
        return new_position
    
    def _calc_new_velocity(self, particle, gbest_pos):
        """Calculate new velocity for a particle."""
        # Maintain some of the current velocity
        new_velocity = particle['velocity'].copy() if random.random() < self.w else []
        
        # Add cognitive component (towards personal best)
        if random.random() < self.c1:
            p_temp = particle['position'].copy()
            for i in range(len(p_temp)):
                if p_temp[i] != particle['pbest_pos'][i]:
                    j = p_temp.index(particle['pbest_pos'][i])
                    new_velocity.append((i, j))
                    p_temp[i], p_temp[j] = p_temp[j], p_temp[i]
        
        # Add social component (towards global best)
        if random.random() < self.c2:
            p_temp = particle['position'].copy()
            for i in range(len(p_temp)):
                if p_temp[i] != gbest_pos[i]:
                    j = p_temp.index(gbest_pos[i])
                    new_velocity.append((i, j))
                    p_temp[i], p_temp[j] = p_temp[j], p_temp[i]
        
        return new_velocity
    
    def solve(self):
        """Run the PSO algorithm."""
        start_time = time.time()
        
        # Initialize particles
        particles = self._create_particles()
        
        # Initialize global best
        gbest_val = float('inf')
        gbest_pos = None
        
        for particle in particles:
            if particle['pbest_val'] < gbest_val:
                gbest_val = particle['pbest_val']
                gbest_pos = particle['pbest_pos'].copy()
        
        for i in tqdm(range(self.iterations), desc="PSO Progress"):
            for j in range(self.num_particles):
                # Calculate new velocity
                particles[j]['velocity'] = self._calc_new_velocity(particles[j], gbest_pos)
                
                # Update position
                particles[j]['position'] = self._apply_velocity(particles[j]['position'], particles[j]['velocity'])
                
                # Evaluate new position
                current_val = self.problem.evaluate_solution(particles[j]['position'])
                
                # Update personal best
                if current_val < particles[j]['pbest_val']:
                    particles[j]['pbest_val'] = current_val
                    particles[j]['pbest_pos'] = particles[j]['position'].copy()
                    
                    # Update global best
                    if current_val < gbest_val:
                        gbest_val = current_val
                        gbest_pos = particles[j]['position'].copy()
            
            # Record the best distance so far
            self.history.append(gbest_val)
        
        end_time = time.time()
        
        return {
            'algorithm': 'Particle Swarm Optimization',
            'solution': gbest_pos,
            'distance': gbest_val,
            'time': end_time - start_time,
            'history': self.history
        }

# Execute the comparison
def main():
    # Set the style for the plots
    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 10))
    
    # Load the data
    print("Loading data...")
    data_file = "data.csv"  # Update this path if needed
    df = load_data(data_file)
    
    # Create the optimization problem
    print("Creating optimization problem...")
    problem = OptimizationProblem(df)
    
    # Run the four algorithms
    algorithms = [
        GeneticAlgorithm(problem, generations=100),
        SimulatedAnnealing(problem, iterations=1000),
        TabuSearch(problem, iterations=1000),
        ParticleSwarmOptimization(problem, iterations=100)
    ]
    
    results = []
    for algo in algorithms:
        print(f"Running {algo.__class__.__name__}...")
        result = algo.solve()
        results.append(result)
        print(f"Best distance: {result['distance']:.2f} km")
        print(f"Execution time: {result['time']:.2f} seconds")
        print("-" * 40)
    
    # Visualize the convergence of algorithms
    plt.figure(figsize=(12, 6))
    for result in results:
        # Normalize iterations for comparison (different algorithms have different numbers of iterations)
        x = np.linspace(0, 1, len(result['history']))
        plt.plot(x, result['history'], label=result['algorithm'])
    
    plt.title('Convergence of Optimization Algorithms', fontsize=15)
    plt.xlabel('Normalized Iterations')
    plt.ylabel('Tour Distance (km)')
    plt.legend()
    plt.grid(True)
    plt.savefig('convergence_comparison.png', dpi=300, bbox_inches='tight')
    
    # Compare final results
    names = [result['algorithm'] for result in results]
    distances = [result['distance'] for result in results]
    times = [result['time'] for result in results]
    
    plt.figure(figsize=(12, 6))
    
    # Plot distances
    plt.subplot(1, 2, 1)
    sns.barplot(x=names, y=distances)
    plt.title('Best Tour Distance (km)')
    plt.xticks(rotation=45)
    plt.ylabel('Distance (km)')
    plt.tight_layout()
    
    # Plot execution times
    plt.subplot(1, 2, 2)
    sns.barplot(x=names, y=times)
    plt.title('Execution Time (seconds)')
    plt.xticks(rotation=45)
    plt.ylabel('Time (s)')
    plt.tight_layout()
    
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    
    # Plot the best routes on a map
    plt.figure(figsize=(15, 10))
    
    for i, result in enumerate(results):
        plt.subplot(2, 2, i+1)
        solution = result['solution']
        
        # Get coordinates of cities in the solution
        lats = [problem.cities.iloc[city]['lat'] for city in solution]
        lngs = [problem.cities.iloc[city]['lng'] for city in solution]
        
        # Close the loop
        lats.append(lats[0])
        lngs.append(lngs[0])
        
        # Plot the route
        plt.plot(lngs, lats, 'b-', alpha=0.5)
        plt.plot(lngs, lats, 'ro', markersize=5)
        
        # Add city names for reference
        for i, city in enumerate(solution):
            plt.annotate(problem.cities.iloc[city]['city'], 
                         (lngs[i], lats[i]),
                         fontsize=8,
                         alpha=0.7)
        
        plt.title(f"{result['algorithm']} - {result['distance']:.2f} km")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('route_comparison.png', dpi=300, bbox_inches='tight')
    
    # Create a comprehensive comparison table
    summary = pd.DataFrame({
        'Algorithm': names,
        'Best Distance (km)': distances,
        'Execution Time (s)': times,
    })
    
    # Add normalized scores (0-100 where 100 is best)
    min_dist = min(distances)
    min_time = min(times)
    
    summary['Distance Score'] = [min_dist / d * 100 for d in distances]
    summary['Speed Score'] = [min_time / t * 100 for t in times]
    summary['Overall Score'] = summary['Distance Score'] * 0.7 + summary['Speed Score'] * 0.3
    
    summary = summary.sort_values('Overall Score', ascending=False)
    print("\nAlgorithm Comparison Summary:")
    print(summary)
    
    # Save the summary table
    summary.to_csv('algorithm_comparison.csv', index=False)
    print("\nVisualizations and summary saved.")

if __name__ == "__main__":
    main()
