import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import networkx as nx
from typing import List, Dict, Tuple, Optional
import random
import time
import copy
from IPython.display import display, clear_output
import os
output_dir = 'results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
class VMICSModelParameters:
    """
    Class to hold all parameters for the VMI-CS model with predictive maintenance
    and transshipment.
    """
    def __init__(self,
                 # Vendor parameters
                 A: float,           # Vendor setup cost
                 h_v: float,         # Vendor holding cost per unit per time
                 P: float,           # Continuous production rate
                 C_in: float,        # Inspection cost per unit
                 C_RW: float,        # Rework cost per unit
                 C_R: float,         # Predictive maintenance cost per unit
                 C_CM: float,        # Corrective maintenance cost per occurrence
                 C_v: float,         # Vendor storage capacity limit
                 X_in: float,        # Percentage of defective items in controlled state
                 X_out: float,       # Percentage of defective items in out-of-control state
                 gamma: float,       # Baseline proportion coefficient for failure rate
                 eta: float,         # Predictive maintenance efficiency coefficient
                 alpha: float,       # Non-linear exponent of predictive maintenance effect

                 # Buyer parameters
                 num_buyers: int,               # Number of buyers
                 A_i: List[float],              # Order costs for buyers
                 h_bi: List[float],             # Buyers' holding costs per unit per time
                 d_i: List[float],              # Buyers' demand rates
                 C_i: List[float],              # Buyers' storage capacity limits
                 s_i: List[float],              # Maximum transportation capacity to buyers

                 # Transshipment parameters
                 A_ij: List[List[float]],       # Transshipment costs between buyers
                 s_ij: List[List[float]]        # Maximum transshipment capacity between buyers
                ):

        # Vendor parameters
        self.A = A
        self.h_v = h_v
        self.P = P
        self.C_in = C_in
        self.C_RW = C_RW
        self.C_R = C_R
        self.C_CM = C_CM
        self.C_v = C_v
        self.X_in = X_in
        self.X_out = X_out
        self.gamma = gamma
        self.eta = eta
        self.alpha = alpha

        # Buyer parameters
        self.num_buyers = num_buyers
        self.A_i = A_i
        self.h_bi = h_bi
        self.d_i = d_i
        self.C_i = C_i
        self.s_i = s_i

        # Transshipment parameters
        self.A_ij = A_ij
        self.s_ij = s_ij

        # Derived parameters
        self.total_demand = sum(d_i)

class VMICSNestedOptimizer:
    """
    Implementation of the nested optimization approach (Algorithm 2) for solving
    the continuous variables (F, T, q_i, q_ij) given fixed discrete variables.
    """
    def __init__(self, params: VMICSModelParameters):
        self.params = params

    def optimize(self, n_i, n_ij):
        """
        Implement Algorithm 2: Nested Optimization for Continuous Variables

        Parameters:
        - n_i: Array of delivery frequencies for each buyer
        - n_ij: Matrix of transshipment indicators between buyers

        Returns:
        - F_star: Optimal predictive maintenance effort
        - T_star: Optimal replenishment cycle length
        - q_i_star: Optimal shipment quantities for each buyer
        - q_ij_star: Optimal transshipment quantities between buyers
        - TC_star: Total cost for the optimal solution
        - is_optimal: Whether the solution satisfies the convexity condition
        """
        # Validate solution: each buyer with n_i = 0 must have incoming transshipments
        for i in range(self.params.num_buyers):
            if n_i[i] == 0:
                has_incoming = False
                for j in range(self.params.num_buyers):
                    if j != i and n_ij[j][i] > 0:
                        has_incoming = True
                        break
                if not has_incoming:
                    # Return very high cost for invalid solution
                    return 0, 0, np.zeros(self.params.num_buyers), np.zeros((self.params.num_buyers, self.params.num_buyers)), float('inf'), False

        # Step 1: Verify convexity condition
        is_optimal = True
        for i in range(self.params.num_buyers):
            # Only check condition when n_i > 0
            if n_i[i] > 0 and self.params.h_v * n_i[i] - self.params.h_bi[i] * n_i[i] * (n_i[i] - 1) < 0:
                is_optimal = False
                # We'll still continue with the optimization but mark the solution as suboptimal

        # Step 2: Define F search range
        F_min = 0.001  # Small positive number
        F_max = (self.params.gamma / self.params.eta) ** (1 / self.params.alpha)

        # Step 3: Apply golden section search for F
        F_star, T_star, q_i_star, q_ij_star, TC_star = self._golden_section_search(
            F_min, F_max, n_i, n_ij, tol=0.0001)

        return F_star, T_star, q_i_star, q_ij_star, TC_star, is_optimal

    def _golden_section_search(self, a, b, n_i, n_ij, tol=0.0001):
        """
        Implement golden section search to find optimal F

        Parameters:
        - a: Lower bound of search interval
        - b: Upper bound of search interval
        - n_i: Array of delivery frequencies for each buyer
        - n_ij: Matrix of transshipment indicators between buyers
        - tol: Tolerance for stopping criterion

        Returns:
        - F_star: Optimal F value
        - T_star, q_i_star, q_ij_star: Corresponding optimal continuous variables
        - TC_star: Corresponding total cost
        """
        # Golden ratio
        gr = (np.sqrt(5) - 1) / 2

        # Initial points
        c = b - gr * (b - a)
        d = a + gr * (b - a)

        # Evaluate function at c and d
        Tc_star, q_ic_star, q_ijc_star, TCc = self._solve_inner_problem(c, n_i, n_ij)
        Td_star, q_id_star, q_ijd_star, TCd = self._solve_inner_problem(d, n_i, n_ij)

        # Store all evaluated solutions
        solutions = [
            (c, Tc_star, q_ic_star, q_ijc_star, TCc),
            (d, Td_star, q_id_star, q_ijd_star, TCd)
        ]

        # Golden section search iterations
        while (b - a) > tol:
            if TCc < TCd:
                b = d
                d = c
                c = b - gr * (b - a)

                # Move results
                TCd = TCc
                Td_star, q_id_star, q_ijd_star = Tc_star, q_ic_star, q_ijc_star

                # Evaluate at new c
                Tc_star, q_ic_star, q_ijc_star, TCc = self._solve_inner_problem(c, n_i, n_ij)
                solutions.append((c, Tc_star, q_ic_star, q_ijc_star, TCc))
            else:
                a = c
                c = d
                d = a + gr * (b - a)

                # Move results
                TCc = TCd
                Tc_star, q_ic_star, q_ijc_star = Td_star, q_id_star, q_ijd_star

                # Evaluate at new d
                Td_star, q_id_star, q_ijd_star, TCd = self._solve_inner_problem(d, n_i, n_ij)
                solutions.append((d, Td_star, q_id_star, q_ijd_star, TCd))

        # Find best solution among all evaluated points
        best_idx = min(range(len(solutions)), key=lambda i: solutions[i][4])
        return solutions[best_idx]

    def _solve_inner_problem(self, F, n_i, n_ij):
        """
        Solve the inner nonlinear programming problem for given F, n_i, n_ij

        Parameters:
        - F: Predictive maintenance effort
        - n_i: Array of delivery frequencies for each buyer
        - n_ij: Matrix of transshipment indicators between buyers

        Returns:
        - T_star: Optimal replenishment cycle length
        - q_i_star: Optimal shipment quantities for each buyer
        - q_ij_star: Optimal transshipment quantities between buyers
        - TC_star: Total cost for the optimal solution
        """
        # Get initial point based on EOQ approximation
        T_0, q_i0, q_ij0 = self._get_initial_point(n_i, n_ij)

        # Define decision variables (T, q_i, q_ij)
        y = self.params.num_buyers
        x0 = [T_0] + list(q_i0) + [q_ij0[i][j] for i in range(y) for j in range(y) if i != j and n_ij[i][j] > 0]

        # Define bounds for variables
        # T > 0, q_i >= 0, q_ij >= 0
        bounds = [(1e-6, None)] + [(0, None)] * (len(x0) - 1)

        # Define constraints
        constraints = self._create_constraints(n_i, n_ij)

        # Define objective function
        def objective(x):
            return self._calculate_total_cost(x, F, n_i, n_ij)

        # Solve nonlinear programming problem
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        # Extract optimal solution
        T_star = result.x[0]
        q_i_star = result.x[1:y+1]

        # Extract q_ij_star based on which n_ij are 1
        q_ij_star = np.zeros((y, y))
        idx = y + 1
        for i in range(y):
            for j in range(y):
                if i != j and n_ij[i][j] > 0:
                    q_ij_star[i, j] = result.x[idx]
                    idx += 1

        # Calculate total cost
        TC_star = result.fun

        return T_star, q_i_star, q_ij_star, TC_star

    def _get_initial_point(self, n_i, n_ij):
        """
        Generate initial point for nonlinear programming based on EOQ approximation

        Parameters:
        - n_i: Array of delivery frequencies for each buyer
        - n_ij: Matrix of transshipment indicators between buyers

        Returns:
        - T_0: Initial replenishment cycle length
        - q_i0: Initial shipment quantities for each buyer
        - q_ij0: Initial transshipment quantities between buyers
        """
        # Calculate initial T using EOQ approximation
        avg_holding_cost = sum(self.params.h_bi) / self.params.num_buyers
        T_0 = np.sqrt(2 * self.params.A / (self.params.total_demand * avg_holding_cost))

        # Calculate initial q_i
        q_i0 = np.zeros(self.params.num_buyers)
        for i in range(self.params.num_buyers):
            # For buyers with n_i > 0, calculate q_i based on demand and frequency
            if n_i[i] > 0:
                q_i0[i] = self.params.d_i[i] * T_0 / n_i[i]
            # For buyers with n_i = 0, q_i should be 0 as they'll receive goods via transshipment

        # Initialize q_ij to a small value for transshipment links
        q_ij0 = np.zeros((self.params.num_buyers, self.params.num_buyers))
        
        # Initialize transshipment quantities for buyers that will receive via transshipment
        for i in range(self.params.num_buyers):
            if n_i[i] == 0:  # This buyer needs to receive via transshipment
                # Find all suppliers
                suppliers = [j for j in range(self.params.num_buyers) if j != i and n_ij[j][i] > 0]
                if suppliers:
                    # Evenly distribute demand among suppliers for initial point
                    amount_per_supplier = self.params.d_i[i] * T_0 / len(suppliers)
                    for j in suppliers:
                        q_ij0[j, i] = amount_per_supplier

        return T_0, q_i0, q_ij0

    def _create_constraints(self, n_i, n_ij):
        """
        Create constraints for the nonlinear programming problem

        Parameters:
        - n_i: Array of delivery frequencies for each buyer
        - n_ij: Matrix of transshipment indicators between buyers

        Returns:
        - constraints: List of constraint dictionaries for scipy.optimize.minimize
        """
        y = self.params.num_buyers

        constraints = []

        # Demand satisfaction constraints
        # n_i*q_i + sum(n_ki*q_ki) - sum(n_ik*q_ik) = d_i*T for all i
        for i in range(y):
            def demand_constraint(x, i=i):
                T = x[0]
                q_i = x[1:y+1]
                q_ij = self._extract_q_ij(x, n_i, n_ij)

                inflow = n_i[i] * q_i[i]
                for k in range(y):
                    if k != i and n_ij[k][i] > 0:
                        inflow += n_ij[k][i] * q_ij[k, i]

                outflow = 0
                for k in range(y):
                    if k != i and n_ij[i][k] > 0:
                        outflow += n_ij[i][k] * q_ij[i, k]

                return inflow - outflow - self.params.d_i[i] * T

            constraints.append({'type': 'eq', 'fun': demand_constraint})

        # Buyer inventory capacity constraints
        # n_i*q_i - (n_i-1)*q_i*d_i/P + sum(n_ki*q_ki) - sum(n_ik*q_ik) <= C_i for all i
        for i in range(y):
            def capacity_constraint(x, i=i):
                T = x[0]
                q_i = x[1:y+1]
                q_ij = self._extract_q_ij(x, n_i, n_ij)

                inventory = n_i[i] * q_i[i]
                if n_i[i] > 1:
                    inventory -= (n_i[i] - 1) * q_i[i] * self.params.d_i[i] / self.params.P

                for k in range(y):
                    if k != i and n_ij[k][i] > 0:
                        inventory += n_ij[k][i] * q_ij[k, i]

                for k in range(y):
                    if k != i and n_ij[i][k] > 0:
                        inventory -= n_ij[i][k] * q_ij[i, k]

                return self.params.C_i[i] - inventory

            constraints.append({'type': 'ineq', 'fun': capacity_constraint})

        # Transshipment feasibility constraints
        # q_i + sum(n_ki*q_ki) - sum(n_ik*q_ik) >= q_i*d_i/P for all i
        for i in range(y):
            def transshipment_constraint(x, i=i):
                q_i = x[1:y+1]
                q_ij = self._extract_q_ij(x, n_i, n_ij)

                inventory = q_i[i]

                for k in range(y):
                    if k != i and n_ij[k][i] > 0:
                        inventory += n_ij[k][i] * q_ij[k, i]

                for k in range(y):
                    if k != i and n_ij[i][k] > 0:
                        inventory -= n_ij[i][k] * q_ij[i, k]

                return inventory - q_i[i] * self.params.d_i[i] / self.params.P

            constraints.append({'type': 'ineq', 'fun': transshipment_constraint})

        # Capacity constraints
        # q_i <= C_v, q_i <= s_i, q_ij <= s_ij for all i, j

        # Vendor capacity constraints
        for i in range(y):
            def vendor_capacity_constraint(x, i=i):
                q_i = x[1:y+1]
                return self.params.C_v - q_i[i]

            constraints.append({'type': 'ineq', 'fun': vendor_capacity_constraint})

        # Transportation capacity constraints
        for i in range(y):
            def transport_capacity_constraint(x, i=i):
                q_i = x[1:y+1]
                return self.params.s_i[i] - q_i[i]

            constraints.append({'type': 'ineq', 'fun': transport_capacity_constraint})

        # Transshipment capacity constraints
        for i in range(y):
            for j in range(y):
                if i != j and n_ij[i][j] > 0:
                    def ts_capacity_constraint(x, i=i, j=j):
                        q_ij = self._extract_q_ij(x, n_i, n_ij)
                        # Use list indexing for s_ij
                        return self.params.s_ij[i][j] - q_ij[i, j]

                    constraints.append({'type': 'ineq', 'fun': ts_capacity_constraint})

        return constraints

    def _extract_q_ij(self, x, n_i, n_ij):
        """
        Extract q_ij values from decision vector x

        Parameters:
        - x: Decision vector
        - n_i: Array of delivery frequencies for each buyer
        - n_ij: Matrix of transshipment indicators between buyers

        Returns:
        - q_ij: Matrix of transshipment quantities
        """
        y = self.params.num_buyers
        q_ij = np.zeros((y, y))

        idx = y + 1
        for i in range(y):
            for j in range(y):
                if i != j and n_ij[i][j] > 0:
                    q_ij[i, j] = x[idx]
                    idx += 1

        return q_ij

    def _calculate_total_cost(self, x, F, n_i, n_ij):
        """
        Calculate total cost for given decision variables

        Parameters:
        - x: Decision vector (T, q_i, q_ij)
        - F: Predictive maintenance effort
        - n_i: Array of delivery frequencies for each buyer
        - n_ij: Matrix of transshipment indicators between buyers

        Returns:
        - TC: Total cost
        """
        y = self.params.num_buyers
        T = x[0]
        q_i = x[1:y+1]
        q_ij = self._extract_q_ij(x, n_i, n_ij)

        # Calculate lambda(F)
        lambda_F = self._calculate_lambda(F)

        # Vendor's basic costs
        TC_v_basic = self.params.A / T + (self.params.h_v / (2 * self.params.P * T)) * sum(n_i[i] * q_i[i]**2 for i in range(y))

        # Vendor's maintenance and quality-related costs
        TC_v_maintenance = (
            self.params.C_in * sum(self.params.d_i) +
            self.params.C_RW * sum(self.params.d_i) * (
                self.params.X_in +
                (self.params.X_out - self.params.X_in) * lambda_F * T * sum(self.params.d_i) / (2 * self.params.P)
            ) +
            self.params.C_R * F * sum(self.params.d_i) / self.params.P +
            self.params.C_CM * sum(self.params.d_i) * lambda_F / self.params.P * (
                1 - lambda_F * T * sum(self.params.d_i) / (2 * self.params.P)
            )
        )

        # Buyers' ordering costs
        OC_b = sum(n_i[i] * self.params.A_i[i] for i in range(y)) / T
        for i in range(y):
            for j in range(y):
                if i != j and n_ij[j][i] > 0:
                    # Use list indexing for A_ij
                    OC_b += n_ij[j][i] * self.params.A_ij[j][i] / T

        # Buyers' inventory holding costs
        HC_b = 0
        for i in range(y):
            inflow = n_i[i] * q_i[i]
            for k in range(y):
                if k != i and n_ij[k][i] > 0:
                    inflow += n_ij[k][i] * q_ij[k, i]

            outflow = 0
            for k in range(y):
                if k != i and n_ij[i][k] > 0:
                    outflow += n_ij[i][k] * q_ij[i, k]

            adjustment = 0
            if n_i[i] > 1:
                adjustment = q_i[i]**2 * n_i[i] * (n_i[i] - 1) / (T * self.params.P)

            HC_b += self.params.h_bi[i] * (inflow - outflow - adjustment) / 2

        # Total cost
        TC = TC_v_basic + TC_v_maintenance + OC_b + HC_b

        return TC

    def _calculate_lambda(self, F):
        """
        Calculate lambda(F) = gamma - eta * F^alpha

        Parameters:
        - F: Predictive maintenance effort

        Returns:
        - lambda_F: Average failure rate
        """
        lambda_F = self.params.gamma - self.params.eta * F**self.params.alpha

        # Ensure lambda is in valid range [0, 1]
        lambda_F = max(0, min(1, lambda_F))

        return lambda_F

class VMICSGeneticAlgorithm:
    """
    Implementation of the genetic algorithm (outer level of the three-level nested optimization)
    for solving the discrete variables (n_i and n_ij).
    """
    def __init__(self, params: VMICSModelParameters, pop_size: int = 50, max_gen: int = 100):
        self.params = params
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.n_genes = params.num_buyers + (params.num_buyers**2 - params.num_buyers)  # n_i + n_ij
        self.best_solution = None
        self.best_fitness = float('-inf')

    def initialize_population(self):
        """Initialize population with specialized and random solutions"""
        population = []

        # Add MST solution
        mst_solution = self._create_mst_solution()
        population.append(mst_solution)

        # Add non-transshipment solution
        non_ts_solution = self._create_non_transshipment_solution()
        population.append(non_ts_solution)

        # Add random solutions
        while len(population) < self.pop_size:
            random_solution = self._create_random_solution()
            # Ensure solution validity - buyers with n_i = 0 must have incoming transshipments
            if self._is_valid_solution(random_solution):
                population.append(random_solution)

        return population

    def _is_valid_solution(self, solution):
        """Check if solution is valid - buyers with n_i = 0 must have incoming transshipments"""
        n_i = solution[:self.params.num_buyers]
        n_ij = np.zeros((self.params.num_buyers, self.params.num_buyers), dtype=int)
        
        idx = self.params.num_buyers
        for i in range(self.params.num_buyers):
            for j in range(self.params.num_buyers):
                if i != j:
                    n_ij[i, j] = solution[idx]
                    idx += 1
        
        for i in range(self.params.num_buyers):
            if n_i[i] == 0:
                has_incoming = False
                for j in range(self.params.num_buyers):
                    if j != i and n_ij[j, i] > 0:
                        has_incoming = True
                        break
                if not has_incoming:
                    return False
        
        return True

    def _create_mst_solution(self):
        """Create a solution based on Minimum Spanning Tree"""
        y = self.params.num_buyers
        # Create a graph with buyers as nodes
        G = nx.Graph()

        # Add nodes for each buyer
        for i in range(y):
            G.add_node(i)

        # Add edges between buyers with costs as weights
        for i in range(y):
            for j in range(y):
                if i != j:
                    G.add_edge(i, j, weight=self.params.A_ij[i][j])

        # Compute MST
        mst = nx.minimum_spanning_tree(G)

        # Create solution
        solution = np.zeros(self.n_genes, dtype=int)

        # Set some delivery frequencies (n_i) to 0 to test transshipment-only scenarios
        # Keep at least one buyer with n_i > 0 to ensure feasibility
        zero_count = min(y-1, int(y/3))  # At most 1/3 of buyers have n_i = 0, but keep at least one with n_i > 0
        zero_buyers = np.random.choice(y, size=zero_count, replace=False)
        
        for i in range(y):
            solution[i] = 0 if i in zero_buyers else np.random.randint(1, 6)

        # Set transshipment indicators (n_ij) based on MST
        idx = y
        for i in range(y):
            for j in range(y):
                if i != j:
                    solution[idx] = 1 if (i, j) in mst.edges or (j, i) in mst.edges else 0
                    idx += 1

        # Ensure buyers with n_i = 0 have incoming transshipments
        for i in range(y):
            if solution[i] == 0:
                has_incoming = False
                for j in range(y):
                    if j != i:
                        idx_ji = y + j * (y - 1) + (i if i < j else i - 1)
                        if solution[idx_ji] == 1:
                            has_incoming = True
                            break
                
                if not has_incoming:
                    # Add a random incoming transshipment
                    j = np.random.choice([j for j in range(y) if j != i])
                    idx_ji = y + j * (y - 1) + (i if i < j else i - 1)
                    solution[idx_ji] = 1

        return solution

    def _create_non_transshipment_solution(self):
        """Create a solution with no transshipments"""
        y = self.params.num_buyers
        solution = np.zeros(self.n_genes, dtype=int)

        # Set delivery frequencies (n_i) to at least 1 for each buyer
        # since there are no transshipments, each buyer must have direct delivery
        for i in range(y):
            solution[i] = max(1, int(np.sqrt(self.params.d_i[i] / sum(self.params.d_i) * y)))

        # Set all transshipment indicators (n_ij) to 0
        # They're already 0 by default, so no need to change

        return solution

    def _create_random_solution(self):
        """Create a random solution"""
        y = self.params.num_buyers
        solution = np.zeros(self.n_genes, dtype=int)

        # Randomly set delivery frequencies (n_i) between 0 and 5
        for i in range(y):
            solution[i] = np.random.randint(0, 6)

        # Randomly set transshipment indicators (n_ij) to 0 or 1
        # With higher probability for 0 (70%)
        for idx in range(y, self.n_genes):
            solution[idx] = 1 if np.random.random() < 0.3 else 0

        # Ensure buyers with n_i = 0 have incoming transshipments
        for i in range(y):
            if solution[i] == 0:
                has_incoming = False
                for j in range(y):
                    if j != i:
                        idx_ji = y + j * (y - 1) + (i if i < j else i - 1)
                        if idx_ji < len(solution) and solution[idx_ji] == 1:
                            has_incoming = True
                            break
                
                if not has_incoming:
                    # Add a random incoming transshipment
                    j = np.random.choice([j for j in range(y) if j != i])
                    idx_ji = y + j * (y - 1) + (i if i < j else i - 1)
                    solution[idx_ji] = 1

        return solution

    def evaluate_fitness(self, chromosome):
        """
        Evaluate fitness of a chromosome using nested optimization
        This calls the middle-level optimization (Algorithm 2)
        """
        nested_optimizer = VMICSNestedOptimizer(self.params)
        discrete_vars = self._decode_chromosome(chromosome)
        
        # Get optimal continuous variables and total cost
        F_star, T_star, q_i_star, q_ij_star, TC_star, is_optimal = nested_optimizer.optimize(
            discrete_vars['n_i'], discrete_vars['n_ij'])

        # Apply penalty if solution is not optimal
        if not is_optimal:
            TC_star *= 1.1  # 10% penalty

        # Return fitness (negative of total cost since we want to maximize fitness)
        return {
            'fitness': -TC_star,
            'F': F_star,
            'T': T_star,
            'q_i': q_i_star,
            'q_ij': q_ij_star,
            'TC': TC_star,
            'is_optimal': is_optimal
        }

    def _decode_chromosome(self, chromosome):
        """Decode chromosome into n_i and n_ij variables"""
        y = self.params.num_buyers

        n_i = chromosome[:y].astype(int)

        n_ij = np.zeros((y, y), dtype=int)
        idx = y
        for i in range(y):
            for j in range(y):
                if i != j:
                    n_ij[i, j] = chromosome[idx]
                    idx += 1

        return {'n_i': n_i, 'n_ij': n_ij}

    def selection(self, population, fitness_values):
        """Select parents using roulette wheel selection"""
        # Adjust fitness values to be positive for roulette wheel
        min_fitness = min(f['fitness'] for f in fitness_values)
        adjusted_fitness = [f['fitness'] - min_fitness + 1e-6 for f in fitness_values]

        total_fitness = sum(adjusted_fitness)
        selection_probs = [f / total_fitness for f in adjusted_fitness]

        # Select parents
        selected_indices = np.random.choice(len(population), size=len(population),
                                           p=selection_probs, replace=True)

        return [population[i] for i in selected_indices]

    def crossover(self, parents):
        """Perform single-point crossover"""
        children = []

        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1 = parents[i]
                parent2 = parents[i + 1]

                if np.random.random() < 0.8:  # 80% crossover probability
                    crossover_point = np.random.randint(1, self.n_genes)
                    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

                    # Fix any invalid children
                    child1 = self._fix_solution(child1)
                    child2 = self._fix_solution(child2)

                    children.append(child1)
                    children.append(child2)
                else:
                    children.append(parent1)
                    children.append(parent2)
            else:
                # If odd number of parents, add the last one
                children.append(parents[i])

        return children

    def _fix_solution(self, solution):
        """Fix solution to ensure buyers with n_i = 0 have incoming transshipments"""
        y = self.params.num_buyers
        n_i = solution[:y]
        
        # Ensure buyers with n_i = 0 have incoming transshipments
        for i in range(y):
            if n_i[i] == 0:
                has_incoming = False
                for j in range(y):
                    if j != i:
                        idx_ji = y + j * (y - 1) + (i if i < j else i - 1)
                        if idx_ji < len(solution) and solution[idx_ji] == 1:
                            has_incoming = True
                            break
                
                if not has_incoming:
                    # Add a random incoming transshipment
                    j = np.random.choice([j for j in range(y) if j != i])
                    idx_ji = y + j * (y - 1) + (i if i < j else i - 1)
                    solution[idx_ji] = 1
        
        return solution

    def mutation(self, children):
        """Apply mutation to children"""
        y = self.params.num_buyers

        for i in range(len(children)):
            for j in range(self.n_genes):
                if np.random.random() < 0.1:  # 10% mutation probability
                    if j < y:  # n_i variables
                        children[i][j] = max(0, children[i][j] + np.random.choice([-1, 1]))
                    else:  # n_ij variables
                        children[i][j] = 1 - children[i][j]  # Flip 0 to 1 or 1 to 0
            
            # Fix any invalid mutation
            children[i] = self._fix_solution(children[i])

        return children

    def run(self):
        """
        Run the genetic algorithm (Implementation of Algorithm 1)

        Returns:
        - best_solution: The best chromosome found
        - best_continuous_vars: Corresponding continuous variables (F_star, T_star, q_i_star, q_ij_star)
        - best_TC: Total cost of the best solution
        """
        # Step 1: Generate initial population
        population = self.initialize_population()

        # Step 2: Evaluate fitness of initial population
        fitness_values = []
        print("Evaluating initial population...")
        for idx, chromosome in enumerate(population):
            fitness_info = self.evaluate_fitness(chromosome)
            fitness_values.append(fitness_info)
            print(f"Chromosome {idx+1}/{len(population)}: TC = {fitness_info['TC']:.2f}, Optimal: {fitness_info['is_optimal']}")

        # Step 3: Initialize algorithm variables
        g = 0
        best_idx = max(range(len(fitness_values)), key=lambda i: fitness_values[i]['fitness'])
        best_chromosome = population[best_idx].copy()
        best_fitness_info = fitness_values[best_idx]
        stagnation_count = 0

        print(f"\nInitial best solution: TC = {best_fitness_info['TC']:.2f}")

        # For tracking progress
        generation_history = []
        best_tc_history = []

        # Step 4: Main loop
        while stagnation_count < 10 and g < self.max_gen:
            g += 1
            print(f"\nGeneration {g}/{self.max_gen} (Stagnation: {stagnation_count}/10)")

            # a. Select parents
            parents = self.selection(population, fitness_values)

            # b. Perform crossover
            offspring = self.crossover(parents)

            # c. Apply mutation
            offspring = self.mutation(offspring)

            # d. Evaluate offspring
            offspring_fitness = []
            for idx, chromosome in enumerate(offspring):
                fitness_info = self.evaluate_fitness(chromosome)
                offspring_fitness.append(fitness_info)
                print(f"Offspring {idx+1}/{len(offspring)}: TC = {fitness_info['TC']:.2f}, Optimal: {fitness_info['is_optimal']}")

            # e. Combine parents and offspring, select best individuals
            combined_population = population + offspring
            combined_fitness = fitness_values + offspring_fitness

            # Sort by fitness (descending)
            sorted_indices = sorted(range(len(combined_fitness)),
                                   key=lambda i: combined_fitness[i]['fitness'],
                                   reverse=True)

            # Select best individuals
            new_population = []
            new_fitness = []
            for i in range(self.pop_size):
                idx = sorted_indices[i]
                new_population.append(combined_population[idx])
                new_fitness.append(combined_fitness[idx])

            population = new_population
            fitness_values = new_fitness

            # f. Check if best solution improved
            best_gen_idx = max(range(len(fitness_values)), key=lambda i: fitness_values[i]['fitness'])
            current_best_fitness_info = fitness_values[best_gen_idx]

            # Record history
            generation_history.append(g)
            best_tc_history.append(current_best_fitness_info['TC'])

            # Check if solution improved by at least 0.001%
            improvement = (best_fitness_info['TC'] - current_best_fitness_info['TC']) / best_fitness_info['TC']
            if improvement >= 0.00001:  # 0.001%
                best_chromosome = population[best_gen_idx].copy()
                best_fitness_info = current_best_fitness_info
                stagnation_count = 0
                print(f"New best solution found: TC = {best_fitness_info['TC']:.2f}, Improvement: {improvement*100:.6f}%")
            else:
                stagnation_count += 1
                print(f"No significant improvement. Best TC = {best_fitness_info['TC']:.2f}")

        # Step 5: Return the best solution
        self.best_solution = best_chromosome
        self.best_fitness = best_fitness_info

        discrete_vars = self._decode_chromosome(best_chromosome)

        result = {
            'discrete_vars': discrete_vars,
            'F': best_fitness_info['F'],
            'T': best_fitness_info['T'],
            'q_i': best_fitness_info['q_i'],
            'q_ij': best_fitness_info['q_ij'],
            'TC': best_fitness_info['TC'],
            'is_optimal': best_fitness_info['is_optimal'],
            'generation_history': generation_history,
            'best_tc_history': best_tc_history
        }

        return result

class VMICSModel:
    """
    Main class for the VMI-CS model with predictive maintenance and transshipment.
    This class orchestrates the overall optimization process.
    """
    def __init__(self, params: VMICSModelParameters, pop_size: int = 50, max_gen: int = 100):
        self.params = params
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.result = None

    def optimize(self):
        """
        Run the full optimization process.

        Returns:
        - result: Dictionary containing the best solution and statistics
        """
        print("Starting VMI-CS optimization with predictive maintenance and transshipment...")
        print(f"Number of buyers: {self.params.num_buyers}")
        print(f"Population size: {self.pop_size}")
        print(f"Maximum generations: {self.max_gen}")
        print(f"Allowing n_i = 0 (buyers can be supplied entirely through transshipment)")

        start_time = time.time()

        # Initialize and run genetic algorithm
        ga = VMICSGeneticAlgorithm(self.params, self.pop_size, self.max_gen)
        self.result = ga.run()

        end_time = time.time()

        self.result['execution_time'] = end_time - start_time

        print("\nOptimization completed.")
        print(f"Execution time: {self.result['execution_time']:.2f} seconds")
        print(f"Total system cost: {self.result['TC']:.2f}")

        return self.result

    def plot_convergence(self):
        """Plot the convergence history of the genetic algorithm"""
        if self.result is None:
            print("No optimization result available. Run optimize() first.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.result['generation_history'], self.result['best_tc_history'], marker='o', linestyle='-')
        plt.title('Convergence of Total Cost')
        plt.xlabel('Generation')
        plt.ylabel('Total Cost')
        plt.grid(True)
        plt.show()

    def plot_network(self):
        """Plot the supply chain network with transshipment"""
        if self.result is None:
            print("No optimization result available. Run optimize() first.")
            return

        y = self.params.num_buyers
        n_i = self.result['discrete_vars']['n_i']
        n_ij = self.result['discrete_vars']['n_ij']
        q_i = self.result['q_i']
        q_ij = self.result['q_ij']

        G = nx.DiGraph()

        # Add vendor node
        G.add_node('Vendor', pos=(0, 0))

        # Add buyer nodes in a circle around the vendor
        buyer_positions = {}
        for i in range(y):
            angle = 2 * np.pi * i / y
            pos = (3 * np.cos(angle), 3 * np.sin(angle))
            G.add_node(f'Buyer {i+1}', pos=pos)
            buyer_positions[f'Buyer {i+1}'] = pos

        # Add vendor-to-buyer edges
        for i in range(y):
            if n_i[i] > 0:
                G.add_edge('Vendor', f'Buyer {i+1}',
                           shipments=n_i[i],
                           quantity=q_i[i],
                           weight=1 + q_i[i]/100)  # Weight affects edge width

        # Add buyer-to-buyer edges (transshipments)
        for i in range(y):
            for j in range(y):
                if i != j and n_ij[i, j] > 0:
                    G.add_edge(f'Buyer {i+1}', f'Buyer {j+1}',
                              shipments=n_ij[i, j],
                              quantity=q_ij[i, j],
                              weight=0.5 + q_ij[i, j]/200)  # Weight affects edge width

        # Prepare the plot
        plt.figure(figsize=(12, 10))

        # Get node positions
        pos = nx.get_node_attributes(G, 'pos')

        # Draw nodes with different colors for buyers with n_i = 0
        node_colors = []
        for node in G.nodes():
            if node == 'Vendor':
                node_colors.append('red')
            else:
                buyer_idx = int(node.split(' ')[1]) - 1
                node_colors.append('lightgreen' if n_i[buyer_idx] == 0 else 'lightblue')

        nx.draw_networkx_nodes(G, pos,
                              node_size=700,
                              node_color=node_colors,
                              edgecolors='black')

        # Draw vendor-to-buyer edges
        vendor_edges = [(u, v) for u, v in G.edges() if u == 'Vendor']
        vendor_edge_widths = [G[u][v]['weight'] * 2 for u, v in vendor_edges]
        nx.draw_networkx_edges(G, pos,
                              edgelist=vendor_edges,
                              width=vendor_edge_widths,
                              edge_color='blue',
                              arrows=True,
                              arrowstyle='-|>',
                              arrowsize=20)

        # Draw buyer-to-buyer edges (transshipments)
        ts_edges = [(u, v) for u, v in G.edges() if u != 'Vendor']
        ts_edge_widths = [G[u][v]['weight'] * 2 for u, v in ts_edges]
        nx.draw_networkx_edges(G, pos,
                              edgelist=ts_edges,
                              width=ts_edge_widths,
                              edge_color='green',
                              style='dashed',
                              arrows=True,
                              arrowstyle='-|>',
                              arrowsize=15)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_weight='bold')

        # Add edge labels showing n_i and q_i
        edge_labels = {}
        for u, v, d in G.edges(data=True):
            edge_labels[(u, v)] = f"{d['shipments']} × {d['quantity']:.1f}"

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        plt.title('VMI-CS Network with Transshipment\nBuyers with n_i=0 shown in green', fontsize=16)
        plt.axis('off')
        plt.show()

    def print_detailed_results(self):
        """Print detailed results of the optimization"""
        if self.result is None:
            print("No optimization result available. Run optimize() first.")
            return

        print("\n============ DETAILED RESULTS ============")
        print(f"Total system cost: {self.result['TC']:.2f}")
        print(f"Predictive maintenance effort (F): {self.result['F']:.4f}")
        print(f"Replenishment cycle length (T): {self.result['T']:.4f}")
        print(f"Average failure rate (λ): {self._calculate_lambda(self.result['F']):.6f}")
        print("\nDelivery frequencies (n_i):")
        n_i = self.result['discrete_vars']['n_i']
        for i in range(len(n_i)):
            if n_i[i] == 0:
                print(f"  Buyer {i+1}: {n_i[i]} (transshipment-only)")
            else:
                print(f"  Buyer {i+1}: {n_i[i]}")

        print("\nDelivery quantities (q_i):")
        q_i = self.result['q_i']
        for i in range(len(q_i)):
            print(f"  Buyer {i+1}: {q_i[i]:.2f}")

        print("\nTransshipment indicators (n_ij) and quantities (q_ij):")
        n_ij = self.result['discrete_vars']['n_ij']
        q_ij = self.result['q_ij']

        transshipments_exist = False
        for i in range(len(n_ij)):
            for j in range(len(n_ij[i])):
                if i != j and n_ij[i, j] > 0:
                    transshipments_exist = True
                    print(f"  From Buyer {i+1} to Buyer {j+1}: {n_ij[i, j]} × {q_ij[i, j]:.2f}")

        if not transshipments_exist:
            print("  No transshipments in the optimal solution.")

        # Calculate cost components
        vendor_basic_cost, vendor_maintenance_cost, buyers_cost = self._calculate_cost_components()

        print("\nCost Breakdown:")
        print(f"  Vendor's basic operational costs: {vendor_basic_cost:.2f} ({vendor_basic_cost/self.result['TC']*100:.1f}%)")
        print(f"  Vendor's maintenance and quality-related costs: {vendor_maintenance_cost:.2f} ({vendor_maintenance_cost/self.result['TC']*100:.1f}%)")
        print(f"  Buyers' costs: {buyers_cost:.2f} ({buyers_cost/self.result['TC']*100:.1f}%)")

        print("\nExecution Information:")
        print(f"  Time taken: {self.result['execution_time']:.2f} seconds")
        print(f"  Convexity condition satisfied: {self.result['is_optimal']}")
        print("==========================================")

    def _calculate_lambda(self, F):
        """Calculate lambda(F) = gamma - eta * F^alpha"""
        lambda_F = self.params.gamma - self.params.eta * F**self.params.alpha
        lambda_F = max(0, min(1, lambda_F))  # Ensure lambda is in [0, 1]
        return lambda_F

    def _calculate_cost_components(self):
        """Calculate the breakdown of costs for detailed reporting"""
        # Extract solution values
        F = self.result['F']
        T = self.result['T']
        n_i = self.result['discrete_vars']['n_i']
        n_ij = self.result['discrete_vars']['n_ij']
        q_i = self.result['q_i']
        q_ij = self.result['q_ij']
        lambda_F = self._calculate_lambda(F)
        y = self.params.num_buyers

        # Vendor's basic costs
        setup_cost = self.params.A / T
        inventory_cost = (self.params.h_v / (2 * self.params.P * T)) * sum(n_i[i] * q_i[i]**2 for i in range(y))
        vendor_basic_cost = setup_cost + inventory_cost

        # Vendor's maintenance and quality-related costs
        inspection_cost = self.params.C_in * sum(self.params.d_i)
        rework_cost = (
            self.params.C_RW * sum(self.params.d_i) * (
                self.params.X_in +
                (self.params.X_out - self.params.X_in) * lambda_F * T * sum(self.params.d_i) / (2 * self.params.P)
            )
        )
        pdm_cost = self.params.C_R * F * sum(self.params.d_i) / self.params.P
        cm_cost = (
            self.params.C_CM * sum(self.params.d_i) * lambda_F / self.params.P * (
                1 - lambda_F * T * sum(self.params.d_i) / (2 * self.params.P)
            )
        )
        vendor_maintenance_cost = inspection_cost + rework_cost + pdm_cost + cm_cost

        # Buyers' costs
        order_cost = sum(n_i[i] * self.params.A_i[i] for i in range(y)) / T
        ts_order_cost = 0
        for i in range(y):
            for j in range(y):
                if i != j and n_ij[j, i] > 0:
                    # Use list indexing for A_ij
                    ts_order_cost += n_ij[j, i] * self.params.A_ij[j][i] / T

        holding_cost = 0
        for i in range(y):
            inflow = n_i[i] * q_i[i]
            for k in range(y):
                if k != i and n_ij[k, i] > 0:
                    inflow += n_ij[k, i] * q_ij[k, i]

            outflow = 0
            for k in range(y):
                if k != i and n_ij[i, k] > 0:
                    outflow += n_ij[i, k] * q_ij[i, k]

            adjustment = 0
            if n_i[i] > 1:
                adjustment = q_i[i]**2 * n_i[i] * (n_i[i] - 1) / (T * self.params.P)

            holding_cost += self.params.h_bi[i] * (inflow - outflow - adjustment) / 2

        buyers_cost = order_cost + ts_order_cost + holding_cost

        return vendor_basic_cost, vendor_maintenance_cost, buyers_cost

def create_example_problem():
    """Create an example problem to test the implementation"""
    # Vendor parameters
    A = 100         # Vendor setup cost
    h_v = 6.0       # Vendor holding cost per unit per time
    P = 5000        # Continuous production rate
    C_in = 0.5      # Inspection cost per unit
    C_RW = 2.0      # Rework cost per unit
    C_R = 25.0      # Predictive maintenance cost per unit
    C_CM = 1000     # Corrective maintenance cost per occurrence
    C_v = 10000      # Vendor storage capacity limit
    X_in = 0.02     # Percentage of defective items in controlled state (2%)
    X_out = 0.25     # Percentage of defective items in out-of-control state (25%)
    gamma = 0.07     # Baseline proportion coefficient for failure rate
    eta = 0.05      # Predictive maintenance efficiency coefficient
    alpha = 0.5    # Non-linear exponent of predictive maintenance effect

    # Buyer parameters
    num_buyers = 3  # Number of buyers

    # Order costs for buyers
    A_i = [130, 80, 120]

    # Buyers' holding costs per unit per time
    h_bi = [4.5, 3.5, 4]

    # Buyers' demand rates
    d_i = [500, 2000, 400]

    # Buyers' storage capacity limits
    C_i = [1000, 1000, 1000]

    # Maximum transportation capacity to buyers
    s_i = [1000, 1000, 1000]

    # Transshipment costs between buyers
    A_ij = [
        [0, 40, 80],
        [40, 0, 40],
        [80, 40, 0]
    ]

    # Maximum transshipment capacity between buyers
    s_ij = [
        [0, 1000, 1000],
        [1000, 0, 1000],
        [1000, 1000, 0]
    ]

    # Create and return model parameters
    params = VMICSModelParameters(
        A=A, h_v=h_v, P=P, C_in=C_in, C_RW=C_RW, C_R=C_R, C_CM=C_CM, C_v=C_v,
        X_in=X_in, X_out=X_out, gamma=gamma, eta=eta, alpha=alpha,
        num_buyers=num_buyers, A_i=A_i, h_bi=h_bi, d_i=d_i, C_i=C_i, s_i=s_i,
        A_ij=A_ij, s_ij=s_ij
    )

    return params

def run_experiment():
    """Run an experiment with the VMI-CS model"""
    # Create example problem
    params = create_example_problem()

    # Create model with reduced population size and generations for quicker demonstration
    model = VMICSModel(params, pop_size=20, max_gen=100)

    # Run optimization
    model.optimize()

    # Print detailed results
    model.print_detailed_results()

    # Plot convergence history
    model.plot_convergence()

    # Plot network
    model.plot_network()

    return model

# If this script is run directly, execute the experiment
if __name__ == "__main__":
    model = run_experiment()