# Copyright 2024 Anshuk Kumar
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from qiskit import QuantumCircuit, Aer
from qiskit_algorithms import QAOA  # Updated import
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import QuantumInstance
from qiskit.quantum_info import Pauli
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators import Operator, Pauli

class TSPComparer:
    def __init__(self):
        # Distance matrix with updated A to C distance
        self.distances = np.array([
            [0, 2, 4],  # A to [A, B, C]
            [2, 0, 1],  # B to [A, B, C]
            [4, 1, 0]   # C to [A, B, C]
        ])
        self.cities = ['A', 'B', 'C']
        self.n_cities = len(self.cities)
        
    def solve_with_brute_force(self):
        """
        Solve TSP using brute force to find the guaranteed optimal solution.
        """
        min_distance = float('inf')
        best_path = None
        
        # Try all possible permutations
        for perm in permutations(range(self.n_cities)):
            distance = 0
            path = list(perm) + [perm[0]]  # Add start city at end
            
            # Calculate total distance
            for i in range(len(path)-1):
                distance += self.distances[path[i]][path[i+1]]
            
            if distance < min_distance:
                min_distance = distance
                best_path = path
        
        return [self.cities[i] for i in best_path], min_distance

    def solve_with_dijkstra(self):
        """
        Solve TSP using Dijkstra's algorithm approach.
        """
        G = nx.Graph()
        for i in range(self.n_cities):
            for j in range(i+1, self.n_cities):
                G.add_edge(self.cities[i], self.cities[j], 
                          weight=self.distances[i][j])
        
        min_distance = float('inf')
        best_path = None
        
        # Try each starting city
        for start in self.cities:
            other_cities = [city for city in self.cities if city != start]
            
            # Try each permutation of remaining cities
            for perm in permutations(other_cities):
                path = [start] + list(perm) + [start]
                distance = sum(self.distances[self.cities.index(path[i])][self.cities.index(path[i+1])] 
                             for i in range(len(path)-1))
                
                if distance < min_distance:
                    min_distance = distance
                    best_path = path
        
        return best_path, min_distance

    def solve_with_qaoa(self):
        """
        Solve TSP using improved QAOA implementation.
        """
        # Create cost Hamiltonian
        n_qubits = self.n_cities * self.n_cities
        cost_ops = []
        
        # Distance terms
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if i != j:
                    for t in range(self.n_cities):
                        t_next = (t + 1) % self.n_cities
                        qubit1 = i * self.n_cities + t
                        qubit2 = j * self.n_cities + t_next
                        
                        pauli_str = ['I'] * n_qubits
                        pauli_str[qubit1] = 'Z'
                        pauli_str[qubit2] = 'Z'
                        pauli = Pauli(''.join(pauli_str))
                        op = Operator(pauli)
                        cost_ops.append(PauliSumOp.from_operator(op, coeff=self.distances[i][j] / 4))
        
        # Strong penalties for constraints
        penalty = 20.0
        
        # One city per time step
        for t in range(self.n_cities):
            for i, j in combinations(range(self.n_cities), 2):
                qubit1 = i * self.n_cities + t
                qubit2 = j * self.n_cities + t
                pauli_str = ['I'] * n_qubits
                pauli_str[qubit1] = 'Z'
                pauli_str[qubit2] = 'Z'
                pauli = Pauli(''.join(pauli_str))
                op = Operator(pauli)
                cost_ops.append(PauliSumOp.from_operator(op, coeff=penalty))
        
        # Each city visited once
        for i in range(self.n_cities):
            for t1, t2 in combinations(range(self.n_cities), 2):
                qubit1 = i * self.n_cities + t1
                qubit2 = i * self.n_cities + t2
                pauli_str = ['I'] * n_qubits
                pauli_str[qubit1] = 'Z'
                pauli_str[qubit2] = 'Z'
                pauli = Pauli(''.join(pauli_str))
                op = Operator(pauli)
                cost_ops.append(PauliSumOp.from_operator(op, coeff=penalty))
        
        hamiltonian = sum(cost_ops)
        
        # Improved QAOA parameters
        optimizer = COBYLA(maxiter=500)
        quantum_instance = QuantumInstance(
            Aer.get_backend('statevector_simulator'),
            shots=2048,
            seed_simulator=123,
            seed_transpiler=123
        )
        
        qaoa = QAOA(
            optimizer=optimizer,
            quantum_instance=quantum_instance,
            reps=4,
            initial_point=[np.pi/4] * 8
        )
        
        result = qaoa.compute_minimum_eigenvalue(hamiltonian)
        
        # Decode with validation
        path_indices = self._decode_qaoa_result(result)
        path = [self.cities[i] for i in path_indices]
        
        # Calculate total distance
        total_distance = sum(self.distances[path_indices[i]][path_indices[i+1]] 
                           for i in range(len(path_indices)-1))
        
        return path, total_distance

    def _decode_qaoa_result(self, result):
        """
        Improved QAOA result decoder with validation.
        """
        counts = result.eigenstate
        max_prob_state = np.argmax(np.abs(counts))
        binary = format(max_prob_state, f'0{self.n_cities * self.n_cities}b')
        state_matrix = np.array(list(map(int, binary))).reshape(self.n_cities, 
                                                              self.n_cities)
        
        path = []
        used_cities = set()
        
        # Greedy path construction with validation
        for t in range(self.n_cities):
            probs = state_matrix[:, t]
            available = [i for i in range(self.n_cities) if i not in used_cities]
            
            if not available:
                remaining = list(set(range(self.n_cities)) - set(path))
                city = remaining[0] if remaining else path[0]
            else:
                city = max(available, key=lambda x: probs[x])
            
            path.append(city)
            used_cities.add(city)
        
        path.append(path[0])  # Complete the cycle
        return path

    def visualize_path(self, path, title):
        """
        Visualize a TSP solution.
        """
        G = nx.Graph()
        for i in range(self.n_cities):
            for j in range(i+1, self.n_cities):
                G.add_edge(self.cities[i], self.cities[j], 
                          weight=self.distances[i][j])
        
        pos = nx.spring_layout(G)
        plt.figure(figsize=(8, 6))
        
        # Draw nodes and labels
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_labels(G, pos)
        
        # Draw edge weights
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels)
        
        # Draw path
        path_edges = list(zip(path[:-1], path))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                             edge_color='r', width=2)
        
        plt.title(title)
        plt.axis('off')
        plt.show()

def main():
    tsp = TSPComparer()
    
    # Find and visualize optimal solution using brute force
    bf_path, bf_distance = tsp.solve_with_brute_force()
    print(f"\nBrute Force Results (Ground Truth):")
    print(f"Optimal path: {' -> '.join(bf_path)}")
    print(f"Total distance: {bf_distance}")
    tsp.visualize_path(bf_path, "Optimal Solution (Brute Force)")
    
    # Find and visualize Dijkstra's solution
    dijk_path, dijk_distance = tsp.solve_with_dijkstra()
    print(f"\nDijkstra's Algorithm Results:")
    print(f"Optimal path: {' -> '.join(dijk_path)}")
    print(f"Total distance: {dijk_distance}")
    tsp.visualize_path(dijk_path, "Solution using Dijkstra's Algorithm")
    
    # Find and visualize QAOA solution
    qaoa_path, qaoa_distance = tsp.solve_with_qaoa()
    print(f"\nQAOA Results:")
    print(f"Optimal path: {' -> '.join(qaoa_path)}")
    print(f"Total distance: {qaoa_distance}")
    tsp.visualize_path(qaoa_path, "Solution using QAOA")

if __name__ == "__main__":
    main()