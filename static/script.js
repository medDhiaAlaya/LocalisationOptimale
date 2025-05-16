// Initialize the map
const map = L.map('map').setView([20, 0], 2);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

// Color scheme for algorithms
const colors = {
    'Genetic Algorithm': '#3498db',
    'Simulated Annealing': '#e74c3c',
    'Tabu Search': '#2ecc71',
    'Particle Swarm Optimization': '#f1c40f'
};

// Show loading spinner
function showLoading() {
    document.querySelector('.loading-spinner').style.display = 'block';
    document.body.classList.add('loading');
}

// Hide loading spinner
function hideLoading() {
    document.querySelector('.loading-spinner').style.display = 'none';
    document.body.classList.remove('loading');
}

// Show loading state for a specific element
function showElementLoading(element) {
    const spinner = element.querySelector('.loading-spinner-small');
    if (spinner) {
        spinner.style.display = 'block';
    }
    if (element.classList.contains('table-loading')) {
        element.classList.add('loading');
    }
}

// Hide loading state for a specific element
function hideElementLoading(element) {
    const spinner = element.querySelector('.loading-spinner-small');
    if (spinner) {
        spinner.style.display = 'none';
    }
    if (element.classList.contains('table-loading')) {
        element.classList.remove('loading');
    }
}

// Fetch and process data
async function loadData() {
    showLoading();
    try {
        const [comparisonData, citiesData] = await Promise.all([
            fetch('/api/comparison').then(res => res.json()),
            fetch('/api/cities').then(res => res.json())
        ]);

        // Update comparison table
        const tableCard = document.querySelector('.table-loading');
        showElementLoading(tableCard);
        updateComparisonTable(comparisonData);
        hideElementLoading(tableCard);
        
        // Create performance chart
        const performanceCard = document.querySelector('#performanceChart').closest('.card');
        showElementLoading(performanceCard);
        createPerformanceChart(comparisonData);
        hideElementLoading(performanceCard);
        
        // Create convergence chart
        const convergenceCard = document.querySelector('#convergenceChart').closest('.card');
        showElementLoading(convergenceCard);
        createConvergenceChart(comparisonData);
        hideElementLoading(convergenceCard);
        
        // Plot routes on map
        const mapCard = document.querySelector('#map').closest('.card');
        showElementLoading(mapCard);
        plotRoutes(comparisonData, citiesData);
        hideElementLoading(mapCard);

        // Add loaded class to cards for fade-in effect
        document.querySelectorAll('.card').forEach(card => {
            card.classList.add('loaded');
        });
    } catch (error) {
        console.error('Error loading data:', error);
    } finally {
        hideLoading();
    }
}

function updateComparisonTable(data) {
    const tbody = document.querySelector('#comparisonTable tbody');
    tbody.innerHTML = '';

    // Calculate performance score (lower is better)
    const scoredData = data.map(algorithm => {
        // Normalize the metrics
        const maxDistance = Math.max(...data.map(d => d.final_distance));
        const maxTime = Math.max(...data.map(d => d.execution_time));
        const maxIterations = Math.max(...data.map(d => d.convergence_iterations));

        const normalizedDistance = algorithm.final_distance / maxDistance;
        const normalizedTime = algorithm.execution_time / maxTime;
        const normalizedIterations = algorithm.convergence_iterations / maxIterations;

        // Calculate performance score (weighted: 40% distance, 40% time, 20% iterations)
        const performanceScore = (normalizedDistance * 0.4) + (normalizedTime * 0.4) + (normalizedIterations * 0.2);
        
        // Calculate additional complexity metrics
        const timePerIteration = algorithm.execution_time / algorithm.convergence_iterations;
        const convergenceRate = (algorithm.convergence_history[0] - algorithm.convergence_history[algorithm.convergence_history.length - 1]) / algorithm.convergence_history[0];
        const stabilityScore = calculateStabilityScore(algorithm.convergence_history);
        const memoryComplexity = calculateMemoryComplexity(algorithm.algorithm);
        
        return {
            ...algorithm,
            performanceScore,
            timePerIteration,
            convergenceRate,
            stabilityScore,
            memoryComplexity
        };
    });

    // Find the best performing algorithm
    const bestScore = Math.min(...scoredData.map(d => d.performanceScore));

    scoredData.forEach(algorithm => {
        const isBestAlgorithm = algorithm.performanceScore === bestScore;
        const row = document.createElement('tr');
        row.className = isBestAlgorithm ? 'best-algorithm' : '';

        // Calculate complexity rating based on multiple factors
        const complexityRating = calculateComplexityRating(algorithm);
        const complexityClass = complexityRating < 1000 ? 'Low' : 
                              complexityRating < 5000 ? 'Medium' : 'High';

        row.innerHTML = `
            <td>
                <span class="algorithm-name">${algorithm.algorithm}</span>
                ${isBestAlgorithm ? '<span class="best-algorithm-badge">Best Performance</span>' : ''}
            </td>
            <td>
                <span class="metric-value">${algorithm.final_distance.toFixed(2)}</span>
                <span class="metric-unit">km</span>
            </td>
            <td>
                <span class="metric-value">${algorithm.execution_time.toFixed(2)}</span>
                <span class="metric-unit">s</span>
            </td>
            <td>
                <span class="metric-value">${algorithm.convergence_iterations}</span>
                <span class="metric-unit">iterations</span>
            </td>
            <td>
                <span class="metric-value">${(algorithm.timePerIteration * 1000).toFixed(2)}</span>
                <span class="metric-unit">ms/iter</span>
            </td>
            <td>
                <span class="metric-value">${(algorithm.convergenceRate * 100).toFixed(1)}</span>
                <span class="metric-unit">%</span>
            </td>
            <td>
                <span class="metric-value">${algorithm.stabilityScore.toFixed(2)}</span>
                <span class="metric-unit">/10</span>
            </td>
            <td>
                <span class="complexity-badge ${algorithm.memoryComplexity.toLowerCase()}">${algorithm.memoryComplexity}</span>
            </td>
            <td>
                <span class="complexity-badge ${complexityClass.toLowerCase()}">${complexityClass}</span>
            </td>
        `;
        tbody.appendChild(row);
    });
}

// Helper function to calculate stability score
function calculateStabilityScore(convergenceHistory) {
    const differences = [];
    for (let i = 1; i < convergenceHistory.length; i++) {
        differences.push(Math.abs(convergenceHistory[i] - convergenceHistory[i-1]));
    }
    const averageDifference = differences.reduce((a, b) => a + b, 0) / differences.length;
    const maxDifference = Math.max(...differences);
    return 10 - (averageDifference / maxDifference * 10);
}

// Helper function to calculate memory complexity
function calculateMemoryComplexity(algorithm) {
    const complexities = {
        'Genetic Algorithm': 'High',
        'Simulated Annealing': 'Low',
        'Tabu Search': 'Medium',
        'Particle Swarm Optimization': 'Medium'
    };
    return complexities[algorithm] || 'Medium';
}

// Helper function to calculate overall complexity rating
function calculateComplexityRating(algorithm) {
    const timeWeight = 0.3;
    const memoryWeight = 0.2;
    const stabilityWeight = 0.2;
    const convergenceWeight = 0.3;

    const timeScore = algorithm.timePerIteration * 1000; // Convert to ms
    const memoryScore = algorithm.memoryComplexity === 'High' ? 100 : 
                       algorithm.memoryComplexity === 'Medium' ? 50 : 25;
    const stabilityScore = (10 - algorithm.stabilityScore) * 10;
    const convergenceScore = (1 - algorithm.convergenceRate) * 100;

    return (timeScore * timeWeight) +
           (memoryScore * memoryWeight) +
           (stabilityScore * stabilityWeight) +
           (convergenceScore * convergenceWeight);
}

function createPerformanceChart(data) {
    const ctx = document.getElementById('performanceChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.algorithm),
            datasets: [{
                label: 'Final Distance (km)',
                data: data.map(d => d.final_distance),
                backgroundColor: data.map(d => colors[d.algorithm]),
                borderColor: data.map(d => colors[d.algorithm]),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Final Distance Comparison'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function createConvergenceChart(data) {
    const ctx = document.getElementById('convergenceChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data[0].convergence_history.map((_, i) => i + 1),
            datasets: data.map(algorithm => ({
                label: algorithm.algorithm,
                data: algorithm.convergence_history,
                borderColor: colors[algorithm.algorithm],
                fill: false,
                tension: 0.4
            }))
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Convergence History'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Distance (km)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Iteration'
                    }
                }
            }
        }
    });
}

function plotRoutes(comparisonData, citiesData) {
    // Clear existing routes
    map.eachLayer((layer) => {
        if (layer instanceof L.Polyline) {
            map.removeLayer(layer);
        }
    });

    // Get the first 20 cities (most populated)
    const topCities = citiesData.slice(0, 20);

    // Plot routes for each algorithm
    comparisonData.forEach(algorithm => {
        // Create a simple route through the cities
        const route = topCities.map(city => [city.lat, city.lng]);

        // Add the route to the map
        L.polyline(route, {
            color: colors[algorithm.algorithm],
            weight: 3,
            opacity: 0.7
        }).addTo(map);

        // Add markers for cities
        route.forEach((coords, index) => {
            const city = topCities[index];
            L.marker(coords)
                .bindPopup(`${city.city} (${city.country})`)
                .addTo(map);
        });
    });

    // Fit map bounds to show all routes
    const bounds = L.latLngBounds(topCities.map(city => [city.lat, city.lng]));
    map.fitBounds(bounds);
}

// Load data when the page loads
document.addEventListener('DOMContentLoaded', () => {
    // Add initial loading states
    document.querySelectorAll('.card').forEach(card => {
        showElementLoading(card);
    });
    
    // Load data
    loadData();
}); 