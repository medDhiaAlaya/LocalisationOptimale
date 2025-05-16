# Optimal Location

This project implements several optimization algorithms to solve the optimal location problem using different metaheuristic approaches.

## 🚀 Features

- Implementation of four optimization algorithms:
  - Genetic Algorithm
  - Simulated Annealing
  - Tabu Search
  - Particle Swarm Optimization
- Interactive web interface
- Results visualization
- Algorithm performance comparison
- Detailed algorithm documentation

## 📋 Prerequisites

- Python 3.8+
- Flask
- NumPy
- Pandas
- Bootstrap 5
- Chart.js
- Leaflet.js

## 🛠️ Installation

1. Clone the repository:

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Usage

1. Launch the Flask application:
```bash
python app.py
```

2. Access the application in your browser:
```
http://localhost:5000
```

## 📊 Implemented Algorithms

### 1. Genetic Algorithm
- **Time Complexity**: O(g * p * n²)
  - g: number of generations
  - p: population size
  - n: number of cities
- **Stop Conditions**:
  - Maximum number of generations reached
  - Population convergence
  - Optimal solution found

### 2. Simulated Annealing
- **Time Complexity**: O(i * n²)
  - i: number of iterations
  - n: number of cities
- **Stop Conditions**:
  - Minimum temperature reached
  - Maximum number of iterations
  - No significant improvement

### 3. Tabu Search
- **Time Complexity**: O(i * n²)
  - i: number of iterations
  - n: number of cities
- **Stop Conditions**:
  - Maximum number of iterations
  - Tabu list saturated
  - No improvement

### 4. Particle Swarm Optimization
- **Time Complexity**: O(i * p * n²)
  - i: number of iterations
  - p: number of particles
  - n: number of cities
- **Stop Conditions**:
  - Maximum number of iterations
  - Particle convergence
  - Optimal solution found

## 👥 Team

- **Mohamed Dhia Alaya**
- **Hamza Ben Ali**
- **Mohamed Taher**
- **Louay Ghnima**

## 📝 Project Structure

```
optimal-location/
├── algorithms/
│   ├── genetic_algorithm.py
│   ├── simulated_annealing.py
│   ├── tabu_search.py
│   ├── particle_swarm.py
│   └── main.py
├── static/
│   ├── style.css
│   └── script.js
├── templates/
│   ├── index.html
│   └── algorithms.html
├── app.py
├── requirements.txt
└── README.md
```

## 📈 Algorithm Comparison

### Evaluation Criteria
1. **Solution Quality**
   - Total route distance
   - Distance score (0-100)

2. **Performance**
   - Execution time
   - Speed score (0-100)

3. **Overall Score**
   - Weighted average of distance and speed scores

### Best Algorithms
- **Best for Quality**: Genetic Algorithm
- **Best for Speed**: Simulated Annealing
- **Best Overall**: Simulated Annealing

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the project
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

