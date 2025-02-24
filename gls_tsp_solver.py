from qubots.base_optimizer import BaseOptimizer
import time

class GLSTSPSolver(BaseOptimizer):
    """
    Guided Local Search (GLS) TSP Solver with automatic coefficient calculation.

    This solver applies a 2‑opt local search enhanced by Guided Local Search.
    It augments the original TSP cost function with penalties on edges (features)
    to help escape local minima.

    The coefficient a balances the penalty term relative to the objective changes.
    It is computed as the average change in the original cost until the first local minimum,
    divided by the number of features (edges) in the tour.
    
    Parameters:
      time_limit (int): Maximum allowed time in seconds.
      max_iterations (int): Maximum number of GLS iterations.
      lambda_param (float): Multiplier controlling the influence of the penalty term.
    """
    def __init__(self, time_limit=300, lambda_param=0.3):
        self.time_limit = time_limit
        self.lambda_param = lambda_param

    def optimize(self, problem, initial_solution=None, **kwargs):
        start_time = time.time()
        time_limit = kwargs.get('time_limit', self.time_limit)
        lambda_param = self.lambda_param

        n = problem.nb_cities
        dist_matrix = problem.dist_matrix

        # Initialize penalties for each edge (using sorted tuple as key).
        penalties = {}
        for i in range(n):
            for j in range(i+1, n):
                penalties[(i, j)] = 0

        def edge_key(i, j):
            return (min(i, j), max(i, j))

        # Temporarily set a to 1.0; it will be updated after the first local search.
        a = 1.0

        # Augmented cost function: original cost plus a penalty term.
        def augmented_cost(tour):
            cost = problem.evaluate_solution(tour)
            penalty_sum = 0
            for k in range(len(tour) - 1):
                key = edge_key(tour[k], tour[k+1])
                penalty_sum += penalties.get(key, 0)
            return cost + lambda_param * a * penalty_sum

        # 2‑opt local search that optionally records improvements.
        def local_search(tour, record_improvements=False):
            improvements = [] if record_improvements else None
            improved = True
            current = tour
            while improved and (time.time() - start_time < time_limit):
                improved = False
                best_neighbor = current
                best_neighbor_cost = augmented_cost(current)
                # Explore all 2‑opt moves.
                for i in range(1, n - 1):
                    for j in range(i + 1, n):
                        if j - i == 1:  # Skip moves that do nothing.
                            continue
                        neighbor = current[:i] + current[i:j][::-1] + current[j:]
                        cost_neighbor = augmented_cost(neighbor)
                        if cost_neighbor < best_neighbor_cost:
                            best_neighbor = neighbor
                            best_neighbor_cost = cost_neighbor
                            improved = True
                if improved:
                    if record_improvements:
                        improvement = problem.evaluate_solution(current) - problem.evaluate_solution(best_neighbor)
                        improvements.append(improvement)
                    current = best_neighbor
            if record_improvements:
                return current, improvements
            else:
                return current

        # Create an initial solution if none is provided.
        initial_tour = initial_solution if initial_solution is not None else problem.random_solution()

        # --- Phase 1: Determine the coefficient a ---
        # Run local search with no penalties (augmented cost equals original cost) and record improvements.
        current_tour, improvements = local_search(initial_tour, record_improvements=True)
        if improvements and len(improvements) > 0:
            avg_improvement = sum(improvements) / len(improvements)
            # For TSP, each tour has (number of cities) edges; we use (len(tour)-1) as the feature count.
            num_features = len(current_tour) - 1  
            a = avg_improvement / num_features
        else:
            a = 1.0  # Fallback if no improvement was recorded.

        # Redefine augmented_cost to use the newly computed a.
        def augmented_cost(tour):
            cost = problem.evaluate_solution(tour)
            penalty_sum = 0
            for k in range(len(tour) - 1):
                key = edge_key(tour[k], tour[k+1])
                penalty_sum += penalties.get(key, 0)
            return cost + lambda_param * a * penalty_sum

        best_tour = current_tour
        best_cost = problem.evaluate_solution(current_tour)

        # --- Phase 2: Guided Local Search iterations ---
        iteration = 0
        while time.time() - start_time < time_limit:
            # Perform local search (with augmented cost) starting from the current solution.
            current_tour = local_search(current_tour, record_improvements=False)
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
