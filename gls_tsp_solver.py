from qubots.base_optimizer import BaseOptimizer
import time
from functools import lru_cache
import numpy as np
class GLSTSPSolver(BaseOptimizer):
    """
    Enhanced Guided Local Search (GLS) TSP Solver with time-aware local search.
    """
    def __init__(self, time_limit=300, lambda_param=0.2):
        self.time_limit = time_limit
        self.lambda_param = lambda_param

    def nearest_neighbor_solution(self, problem):
        """Generate initial tour using the nearest neighbor heuristic."""
        n = problem.nb_cities
        dist_matrix = problem.dist_matrix
        start = 0
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        current_city = start
        while unvisited:
            next_city = min(unvisited, key=lambda city: dist_matrix[current_city][city])
            tour.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
        return tour

    def optimize(self, problem, initial_solution=None, **kwargs):
        start_time = time.time()
        time_limit = kwargs.get('time_limit', self.time_limit)
        lambda_param = self.lambda_param

        n = problem.nb_cities
        dist_matrix = problem.dist_matrix

        # Initialize edge penalties
        penalties = {}
        for i in range(n):
            for j in range(i + 1, n):
                penalties[(i, j)] = 0

        def edge_key(i, j):
            return (min(i, j), max(i, j))

        a = 1.0  # Initial coefficient

        def augmented_cost(tour):
            """Compute the augmented cost including penalties."""
            cost = problem.evaluate_solution(tour)
            penalty_sum = sum(penalties.get(edge_key(tour[k], tour[k+1]), 0) for k in range(len(tour)-1))
            return cost + lambda_param * a * penalty_sum
        
        def create_neighbor_lists(problem, k=20):
            """Precompute k nearest neighbors for each city"""
            return [
                np.argsort(dist_matrix[i])[1:k+1]  # Exclude self
                for i in range(problem.nb_cities)
            ]
        
        @lru_cache(maxsize=100000)
        def cached_dist(a, b):
            return dist_matrix[a][b]

        def local_search(tour, record_improvements=False):
            """Optimized 2-opt with neighbor lists and first-improvement"""
            current = tour.copy()
            improved = True
            neighbor_list = create_neighbor_lists(problem)  # Precomputed
            
            while improved and (time.time() - start_time < time_limit):
                improved = False
                shuffled_indices = np.random.permutation(len(current)-2) + 1  # Randomized search
                
                for i in shuffled_indices:
                    B = current[i]
                    candidates = [idx for idx, city in enumerate(current) 
                                if city in neighbor_list[B] and idx > i]
                    
                    for j in candidates:
                        A, C, D, D_next = current[i-1], current[j-1], current[j], current[(j+1)%len(current)]
                        
                        # Fast delta calculation
                        delta = (cached_dist(A, C) + cached_dist(B, D_next)) - \
                                (cached_dist(A, B) + cached_dist(C, D))
                        
                        if delta < -1e-6:  # Threshold for numerical stability
                            current = current[:i] + current[i:j][::-1] + current[j:]
                            improved = True
                            break
                    if improved:
                        break
                        
            return current

        # --- Phase 1: Determine coefficient 'a' ---
        initial_tour = self.nearest_neighbor_solution(problem) if initial_solution is None else initial_solution
        #current_tour, improvements = local_search(initial_tour, record_improvements=True)
        #if improvements:
        #    avg_improvement = sum(improvements) / len(improvements)
        #    a = avg_improvement / (len(current_tour) - 1) if len(current_tour) > 1 else 1.0
        #else:
        #    a = 1.0
        a = 1
        # --- Phase 2: GLS iterations ---
        best_tour, best_cost = initial_tour, problem.evaluate_solution(initial_tour)
        current_tour = best_tour.copy()
        stagnation = 0
        print(f"C = {best_cost}")
        while time.time() - start_time < time_limit:
            current_tour = local_search(current_tour)
            current_cost = problem.evaluate_solution(current_tour)
            print(f"CC = {current_cost}")
            if current_cost < best_cost:
                best_tour, best_cost = current_tour, current_cost
                stagnation = 0
            else:
                stagnation += 1
                if stagnation >= 50:
                    break  # Terminate if no improvement in 50 iterations

            # Update penalties based on current tour
            utilities = {}
            max_util = -1
            for k in range(len(current_tour) - 1):
                i, j = current_tour[k], current_tour[k+1]
                key = edge_key(i, j)
                utilities[key] = dist_matrix[i][j] / (1 + penalties[key])
                if utilities[key] > max_util:
                    max_util = utilities[key]

            # Penalize edges with maximum utility
            for key in utilities:
                if abs(utilities[key] - max_util) < 1e-6:
                    penalties[key] += 1

        return best_tour, best_cost