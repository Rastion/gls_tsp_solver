from qubots.base_optimizer import BaseOptimizer
import time

class GLSTSPSolver(BaseOptimizer):
    """
    Guided Local Search (GLS) TSP Solver.

    This solver applies a 2-opt local search enhanced by Guided Local Search.
    It augments the original TSP cost function with penalties on edges (features)
    to help escape local minima.

    Parameters:
      time_limit (int): Maximum allowed time in seconds.
      max_iterations (int): Maximum number of GLS iterations.
      lambda_param (float): Parameter controlling the influence of the penalty term.
    """
    def __init__(self, time_limit=300, lambda_param=0.3):
        self.time_limit = time_limit
        self.lambda_param = lambda_param

    def optimize(self, problem, initial_solution=None, **kwargs):
        start_time = time.time()
        time_limit = kwargs.get('time_limit', self.time_limit)
        max_iterations = kwargs.get('max_iterations', self.max_iterations)
        lambda_param = self.lambda_param

        n = problem.nb_cities
        dist_matrix = problem.dist_matrix

        # Initialize penalties for each edge (for symmetric TSP, use sorted tuple as key)
        penalties = {}
        for i in range(n):
            for j in range(i + 1, n):
                penalties[(i, j)] = 0

        def edge_key(i, j):
            return (min(i, j), max(i, j))

        def original_cost(tour):
            cost = 0
            for k in range(len(tour) - 1):
                cost += dist_matrix[tour[k]][tour[k+1]]
            return cost

        # For the augmented cost, we add lambda * a * (sum of penalties for edges in the tour)
        # For simplicity, we set a = 1.0 by default.
        a = 1.0
        def augmented_cost(tour):
            cost = problem.evaluate_solution(tour)
            penalty_sum = 0
            for k in range(len(tour) - 1):
                key = edge_key(tour[k], tour[k+1])
                penalty_sum += penalties.get(key, 0)
            return cost + lambda_param * a * penalty_sum

        # Generate an initial solution if not provided.
        current_tour = initial_solution if initial_solution is not None else problem.random_solution()
        best_tour = current_tour
        best_cost = problem.evaluate_solution(current_tour)

        # 2-opt local search using the augmented cost function.
        def local_search(tour):
            improved = True
            current = tour
            while improved and time.time() - start_time < time_limit:
                improved = False
                best_neighbor = current
                best_neighbor_cost = augmented_cost(current)
                for i in range(1, n - 1):
                    for j in range(i + 1, n):
                        if j - i == 1:  # Skip consecutive nodes.
                            continue
                        # 2-opt move: reverse the segment between i and j.
                        neighbor = current[:i] + current[i:j][::-1] + current[j:]
                        cost_neighbor = augmented_cost(neighbor)
                        if cost_neighbor < best_neighbor_cost:
                            best_neighbor = neighbor
                            best_neighbor_cost = cost_neighbor
                            improved = True
                current = best_neighbor
            return current

        iteration = 0
        while time.time() - start_time < time_limit:
            # Local search phase
            current_tour = local_search(current_tour)
            current_orig_cost = problem.evaluate_solution(current_tour)
            if current_orig_cost < best_cost:
                best_tour = current_tour
                best_cost = current_orig_cost

            if time.time() - start_time >= time_limit:
                break

            # Compute utilities for each edge in the current tour.
            # Utility = (edge cost) / (1 + current penalty)
            utilities = {}
            max_util = -1
            for k in range(len(current_tour) - 1):
                i, j = current_tour[k], current_tour[k+1]
                key = edge_key(i, j)
                edge_cost = dist_matrix[i][j]
                util = edge_cost / (1 + penalties[key])
                utilities[key] = util
                if util > max_util:
                    max_util = util

            # Increase penalty for all edges with maximum utility.
            for key, util in utilities.items():
                if abs(util - max_util) < 1e-6:
                    penalties[key] += 1

            iteration += 1

        return best_tour, best_cost
