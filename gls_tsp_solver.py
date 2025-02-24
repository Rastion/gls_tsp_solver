from qubots.base_optimizer import BaseOptimizer
import time

class GLSTSPSolver(BaseOptimizer):
    """
    Enhanced Guided Local Search (GLS) TSP Solver with delta evaluations, nearest neighbor initialization, and stagnation detection.
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

        def local_search(tour, record_improvements=False):
            """2-opt local search with delta evaluations."""
            current = tour.copy()
            improved = True
            improvements = []
            original_cost = problem.evaluate_solution(current)
            best_augmented = augmented_cost(current)

            while improved and (time.time() - start_time < time_limit):
                improved = False
                best_delta = 0
                best_move = None

                for i in range(1, len(current)-1):
                    for j in range(i+1, len(current)):
                        if j == i + 1:
                            continue  # Skip adjacent swap

                        # Nodes involved in the 2-opt move
                        A, B, C, D = current[i-1], current[i], current[j-1], current[j]

                        # Delta for original cost
                        delta_original = (dist_matrix[A][C] + dist_matrix[B][D]) - (dist_matrix[A][B] + dist_matrix[C][D])

                        # Delta for penalties
                        edge_AC, edge_BD = edge_key(A, C), edge_key(B, D)
                        edge_AB, edge_CD = edge_key(A, B), edge_key(C, D)
                        delta_penalty = (penalties[edge_AC] + penalties[edge_BD] - penalties[edge_AB] - penalties[edge_CD])

                        # Total delta for augmented cost
                        delta_total = delta_original + lambda_param * a * delta_penalty

                        if delta_total < 0:  # Improving move
                            if not improved or delta_total < best_delta:
                                best_delta = delta_total
                                best_move = (i, j)
                                improved = True

                if improved:
                    # Apply the best move
                    i, j = best_move
                    current = current[:i] + current[i:j][::-1] + current[j:]
                    best_augmented += best_delta
                    if record_improvements:
                        improvements.append(-delta_original)  # Track original cost improvement
                    improved = True

            return (current, improvements) if record_improvements else current

        # Phase 1: Determine coefficient 'a'
        initial_tour = self.nearest_neighbor_solution(problem) if initial_solution is None else initial_solution
        current_tour, improvements = local_search(initial_tour, record_improvements=True)
        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            a = avg_improvement / (len(current_tour) - 1) if len(current_tour) > 1 else 1.0
        else:
            a = 1.0

        # Phase 2: GLS iterations
        best_tour, best_cost = current_tour, problem.evaluate_solution(current_tour)
        stagnation = 0

        while time.time() - start_time < time_limit:
            current_tour = local_search(current_tour)
            current_cost = problem.evaluate_solution(current_tour)

            if current_cost < best_cost:
                best_tour, best_cost = current_tour, current_cost
                stagnation = 0
            else:
                stagnation += 1
                if stagnation >= 50:
                    break  # Stagnation detected

            # Update penalties based on current tour
            utilities = {}
            max_util = -1
            for k in range(len(current_tour) - 1):
                i, j = current_tour[k], current_tour[k+1]
                key = edge_key(i, j)
                utilities[key] = dist_matrix[i][j] / (1 + penalties[key])
                if utilities[key] > max_util:
                    max_util = utilities[key]

            # Apply penalties to edges with max utility
            for key in utilities:
                if abs(utilities[key] - max_util) < 1e-6:
                    penalties[key] += 1

        return best_tour, best_cost