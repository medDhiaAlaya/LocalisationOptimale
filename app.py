from flask import Flask, render_template, jsonify
import pandas as pd
import json
import numpy as np
import os

app = Flask(__name__)

def load_data():
    # Load the comparison data
    comparison_df = pd.read_csv('algorithm_comparison.csv')
    
    # Load the city data
    cities_df = pd.read_csv('data.csv')
    
    # Format the comparison data for the web app
    comparison_data = []
    for _, row in comparison_df.iterrows():
        algorithm_data = {
            'algorithm': row['Algorithm'],
            'final_distance': float(row['Best Distance (km)']),
            'execution_time': float(row['Execution Time (s)']),
            'convergence_iterations': 100,  # Default value since not in CSV
            'convergence_history': [float(row['Best Distance (km)']) * (1 - i/100) for i in range(100)],  # Simulated convergence history
            'time_complexity': row['Time Complexity']  # Add time complexity
        }
        comparison_data.append(algorithm_data)
    
    # Convert cities data to dictionary and handle NaN values
    cities_data = []
    for _, row in cities_df.iterrows():
        city_dict = {}
        for column in cities_df.columns:
            value = row[column]
            if pd.isna(value):
                city_dict[column] = None
            elif isinstance(value, (np.int64, np.float64)):
                city_dict[column] = float(value)
            else:
                city_dict[column] = value
        cities_data.append(city_dict)
    
    return comparison_data, cities_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/algorithms')
def algorithms():
    return render_template('algorithms.html')

@app.route('/api/comparison')
def get_comparison():
    comparison_data, _ = load_data()
    return jsonify(comparison_data)

@app.route('/api/cities')
def get_cities():
    _, cities_data = load_data()
    return jsonify(cities_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 