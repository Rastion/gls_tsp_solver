from qubots.base_optimizer import BaseOptimizer
import time
import numpy as np

class GLSTSPSolver(BaseOptimizer):
    """
    Scalable Guided Local Search for TSP with adaptive neighborhood management.
    Features:
    - Matrix-based penalty tracking
    - Precomputed neighbor lists
    - Time-aware iterative deepening
    - Dynamic neighborhood sizing
    """
    def __init__(self, time_limit=300, lambda_param=0.2, neighbor_ratio=0.05):
        self.time_limit = time_limit
        self.lambda_param = lambda_param
        self.neighbor_ratio = neighbor_ratio

    def nearest_neighbor_solution(self, problem):
        """Vectorized nearest neighbor implementation"""
        n = problem.nb_cities
        dist_matrix = problem.dist_matrix
        tour = [0]
        mask = np.ones(n, dtype=bool)
        mask[0] = False
        
        for _ in range(n-1):
            last = tour[-1]
            candidates = np.where(mask)[0]
            if not candidates.size:
                break
            next_city = candidates[np.argmin(dist_matrix[last, candidates])]
            tour.append(next_city)
            mask[next_city] = False
            
        return tour

    def optimize(self, problem, initial_solution=None, verbose=False, **kwargs):
        start_time = time.time()
        time_limit = kwargs.get('time_limit', self.time_limit)
        n = problem.nb_cities
        dist_matrix = np.asarray(problem.dist_matrix)
        k = max(10, int(n * self.neighbor_ratio))

        # Precompute data structures
        neighbor_lists = np.zeros((n, k), dtype=int)
        for i in range(n):
            neighbor_lists[i] = np.argsort(dist_matrix[i])[1:k+1]
            
        # Penalty matrix (symmetric storage)
        penalties = np.zeros((n, n), dtype=int)

        # Adaptive parameter initialization
        a = 1.0  

        if initial_solution:
            current_tour = initial_solution
        else:
            current_tour = self.nearest_neighbor_solution(problem)

        best_tour = current_tour.copy()
        best_cost = problem.evaluate_solution(best_tour)
        last_improvement = start_time

        # Iterative deepening main loop
        while time.time() - start_time < time_limit:
            # Phase 1: Intensification with time-budgeted 2-opt
            current_tour = self.time_aware_2opt(
                current_tour, dist_matrix, neighbor_lists,
                start_time, time_limit - (time.time() - start_time),
                penalties, a
            )
            
            # Phase 2: Diversification through penalty updates
            current_cost = problem.evaluate_solution(current_tour)
            if verbose:
                print(f"Current cost: {current_cost}")
            if current_cost < best_cost:
                best_tour, best_cost = current_tour, current_cost
                last_improvement = time.time()
            
            # Adaptive penalty activation
            self.update_penalties(current_tour, dist_matrix, penalties)
            
            # Escape condition: No improvement in adaptive window
            if time.time() - last_improvement > (time_limit * 0.2):
                current_tour = self.diversify_solution(best_tour)

        return best_tour, best_cost

    def time_aware_2opt(self, tour, dist_matrix, neighbor_lists, 
                   start_time, time_budget, penalties, a):
        """2-opt with depot locking at position 0"""
        n = len(tour)
        improved = True
        current = np.array(tour)
        #best_cost = self.augmented_cost(current, dist_matrix, penalties, a)
        
        # Only permute indices from 1 to n-2 (never touch depot)
        while improved and (time.time() - start_time < time_budget):
            improved = False
            for i in (np.random.permutation(n-2) + 1):  # Start from index 1
                if time.time() - start_time >= time_budget:
                    return current.tolist()
                    
                B = current[i]
                candidates = neighbor_lists[B]
                in_tour = np.isin(current, candidates)
                possible_j = np.where(in_tour)[0]
                possible_j = possible_j[possible_j > i]
                
                for j in possible_j:
                    if j <= i+1:
                        continue
                        
                    A, B_node = current[i-1], B
                    C, D = current[j-1], current[j]
                    
                    # Skip moves involving depot (index 0)
                    if 0 in {A, B_node, C, D}:
                        continue
                    
                    delta = (dist_matrix[A,C] + dist_matrix[B_node,D]) - \
                            (dist_matrix[A,B_node] + dist_matrix[C,D])
                            
                    penalty_delta = self.lambda_param * a * (
                        penalties[A,C] + penalties[B_node,D] - 
                        penalties[A,B_node] - penalties[C,D]
                    )
                    
                    if delta + penalty_delta < -1e-6:
                        current[i:j] = current[i:j][::-1]
                        improved = True
                        break
                if improved:
                    break
                    
        return current.tolist()

    def update_penalties(self, tour, dist_matrix, penalties):
        """Sparse penalty update focusing on critical edges"""
        utilities = []
        for i in range(len(tour)-1):
            u, v = sorted([tour[i], tour[i+1]])
            utility = dist_matrix[u,v] / (1 + penalties[u,v])
            utilities.append( (utility, u, v) )
            
        max_util = max(utilities, key=lambda x: x[0])[0]
        for util, u, v in utilities:
            if util >= max_util - 1e-6:
                penalties[u,v] += 1

    def augmented_cost(self, tour, dist_matrix, penalties, a):
        """Vectorized augmented cost calculation"""
        edges = np.sort(np.column_stack([tour[:-1], tour[1:]]), axis=1)
        base_cost = dist_matrix[edges[:,0], edges[:,1]].sum()
        penalty_cost = penalties[edges[:,0], edges[:,1]].sum()
        return base_cost + self.lambda_param * a * penalty_cost

    def diversify_solution(self, tour):
        """Double bridge kick with depot protection"""
        n = len(tour)
        if n < 8:
            return tour.copy()
        
        # Ensure perturbations never affect depot
        while True:
            indices = np.random.choice(range(1, n-2), 4, replace=False)
            a, b, c, d = sorted(indices)
            if (a > 0 and d < n-1 and 
                (b-a > 1) and (c-b > 1) and (d-c > 1)):
                break
                
        return tour[:a] + tour[c:d] + tour[b:c] + tour[a:b] + tour[d:]