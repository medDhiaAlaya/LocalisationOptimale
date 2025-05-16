# Localisation Optimale

Ce projet implémente plusieurs algorithmes d'optimisation pour résoudre le problème de localisation optimale, en utilisant différentes approches métaheuristiques.

## 🚀 Fonctionnalités

- Implémentation de quatre algorithmes d'optimisation :
  - Algorithme Génétique
  - Recuit Simulé
  - Recherche Tabou
  - Optimisation par Essaim Particulaire
- Interface web interactive
- Visualisation des résultats
- Comparaison des performances des algorithmes
- Documentation détaillée des algorithmes

## 📋 Prérequis

- Python 3.8+
- Flask
- NumPy
- Pandas
- Bootstrap 5
- Chart.js
- Leaflet.js

## 🛠️ Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/votre-username/localisation-optimale.git
cd localisation-optimale
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Utilisation

1. Lancez l'application Flask :
```bash
python app.py
```

2. Accédez à l'application dans votre navigateur :
```
http://localhost:5000
```

## 📊 Algorithmes Implémentés

### 1. Algorithme Génétique
- **Complexité Temporelle** : O(g * p * n²)
  - g : nombre de générations
  - p : taille de la population
  - n : nombre de villes
- **Conditions d'Arrêt** :
  - Nombre maximum de générations atteint
  - Convergence de la population
  - Solution optimale trouvée

### 2. Recuit Simulé
- **Complexité Temporelle** : O(i * n²)
  - i : nombre d'itérations
  - n : nombre de villes
- **Conditions d'Arrêt** :
  - Température minimale atteinte
  - Nombre maximum d'itérations
  - Pas d'amélioration significative

### 3. Recherche Tabou
- **Complexité Temporelle** : O(i * n²)
  - i : nombre d'itérations
  - n : nombre de villes
- **Conditions d'Arrêt** :
  - Nombre maximum d'itérations
  - Liste taboue saturée
  - Pas d'amélioration

### 4. Optimisation par Essaim Particulaire
- **Complexité Temporelle** : O(i * p * n²)
  - i : nombre d'itérations
  - p : nombre de particules
  - n : nombre de villes
- **Conditions d'Arrêt** :
  - Nombre maximum d'itérations
  - Convergence des particules
  - Solution optimale trouvée

## 👥 Équipe

- **Mohamed Dhia Alaya**
- **Hamza Ben Ali**
- **Mohamed Taher**
- **Louay Ghnima**

## 📝 Structure du Projet

```
localisation-optimale/
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

## 📈 Comparaison des Algorithmes

### Critères d'Évaluation
1. **Qualité de la Solution**
   - Distance totale de l'itinéraire
   - Score de distance (0-100)

2. **Performance**
   - Temps d'exécution
   - Score de vitesse (0-100)

3. **Score Global**
   - Moyenne pondérée des scores de distance et de vitesse

### Meilleurs Algorithmes
- **Meilleur pour la Qualité** : Algorithme Génétique
- **Meilleur pour la Vitesse** : Recuit Simulé
- **Meilleur Global** : Recuit Simulé

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 📞 Contact

Pour toute question ou suggestion, n'hésitez pas à nous contacter à [votre-email@example.com](mailto:votre-email@example.com)
