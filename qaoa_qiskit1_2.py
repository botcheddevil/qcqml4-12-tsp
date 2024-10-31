import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from qiskit_aer import QasmSimulator
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit.opflow import PauliOp
from qiskit.quantum_info import Pauli
from qiskit_ibm_runtime import QiskitRuntimeService

def decode_solution(result, n_cities):
    """
    Decode the QAOA result into a valid TSP path.
    Now properly handles dictionary output from QAOA.
    """
    try:
        # Get the state with highest probability from the counts dictionary
        if hasattr(result, 'eigenstate') and isinstance(result.eigenstate, dict):
            # Handle dictionary of counts
            counts = result.eigenstate
            max_bitstring = max(counts.items(), key=lambda x: x[1])[0]
            binary = format(int(max_bitstring, 2), f'0{n_cities * n_cities}b')
        else:
            # Fallback: create a simple sequential path
            print("Warning: Could not decode quantum state, using fallback path")
            path = list(range(n_cities)) + [0]
            return path

        # Convert to state matrix
        state_matrix = np.array(list(map(int, binary))).reshape(n_cities, n_cities)

        # Build valid path
        path = []
        used_cities = set()

        for t in range(n_cities):
            probs = state_matrix[:, t]
            available = [i for i in range(n_cities) if i not in used_cities]

            if not available:
                remaining = list(set(range(n_cities)) - set(path))
                city = remaining[0] if remaining else path[0]
            else:
                city = max(available, key=lambda x: probs[x])

            path.append(city)
            used_cities.add(city)

        path.append(path[0])  # Complete the cycle
        return path

    except Exception as e:
        print(f"Warning: Error in solution decoding: {e}")
        print("Using fallback path")
        # Return a simple sequential path as fallback
        return list(range(n_cities)) + [0]

def create_cost_hamiltonian(distances):
    """
    Create the cost Hamiltonian for the QAOA.
    """
    n_cities = len(distances)
    n_qubits = n_cities * n_cities
    cost_ops = []

    # Distance terms
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                for t in range(n_cities):
                    t_next = (t + 1) % n_cities
                    qubit1 = i * n_cities + t
                    qubit2 = j * n_cities + t_next

                    pauli_str = ['I'] * n_qubits
                    pauli_str[qubit1] = 'Z'
                    pauli_str[qubit2] = 'Z'
                    pauli = Pauli(''.join(pauli_str))

                    cost_ops.append(PauliOp(pauli, distances[i][j] / 4))

    # Add strong penalty terms
    penalty = 20.0

    # One city per time step
    for t in range(n_cities):
        for i, j in combinations(range(n_cities), 2):
            qubit1 = i * n_cities + t
            qubit2 = j * n_cities + t

            pauli_str = ['I'] * n_qubits
            pauli_str[qubit1] = 'Z'
            pauli_str[qubit2] = 'Z'
            pauli = Pauli(''.join(pauli_str))

            cost_ops.append(PauliOp(pauli, penalty))

    # Each city visited once
    for i in range(n_cities):
        for t1, t2 in combinations(range(n_cities), 2):
            qubit1 = i * n_cities + t1
            qubit2 = i * n_cities + t2

            pauli_str = ['I'] * n_qubits
            pauli_str[qubit1] = 'Z'
            pauli_str[qubit2] = 'Z'
            pauli = Pauli(''.join(pauli_str))

            cost_ops.append(PauliOp(pauli, penalty))

    return sum(cost_ops)

def solve_tsp_with_qaoa(distances, cities, useSimulator=True, visualize=False):
    """
    Solve TSP using QAOA on IBM Quantum hardware.
    """
    # Create problem instance
    n_cities = len(distances)

    print(f"Solving TSP for {n_cities} cities:\n", cities, "\nDistance Matrix:\n", distances)
    if visualize:
        # Show initial graph
        print("Visualizing TSP Graph:")
        visualize_graph(distances, cities)

    # Create Hamiltonian
    cost_hamiltonian = create_cost_hamiltonian(distances)

    # Set up QAOA
    p = 3  # Number of QAOA layers (keep small for hardware constraints)
    optimizer = COBYLA(maxiter=500)

    if useSimulator:
        backend = QasmSimulator()
    else:
        # Save the IBM Quantum Experience Credentials only for first run and DO NOT COMMIT the Token in GIT repo!
        # QiskitRuntimeService.save_account(channel="ibm_quantum", token="<YOUR-TOKEN>", overwrite=True, set_as_default=True)
        
        # Load IBMQ account and select backend
        service = QiskitRuntimeService(channel="ibm_quantum")
        # Get the least busy backend with enough qubits
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=16)
        if not backend:
            raise Exception("No available backends with enough qubits.")
        
        print("Running on current least busy backend:", backend)

    qaoa = QAOA(
        optimizer=optimizer,
        quantum_instance=backend,
        reps=p,
        initial_point=[np.pi/3] * (2*p)
    )

    # Run QAOA
    try:
        result = qaoa.compute_minimum_eigenvalue(cost_hamiltonian)

        # Debug information
        print("\nQAOA Result Details:")
        print(f"Result type: {type(result)}")
        print(f"Available attributes: {dir(result)}")
        if hasattr(result, 'eigenstate'):
            print(f"Eigenstate type: {type(result.eigenstate)}")

        # Decode solution
        path_indices = decode_solution(result, n_cities)
        path = [cities[i] for i in path_indices]

        # Calculate total distance
        total_distance = sum(distances[path_indices[i]][path_indices[i+1]]
                           for i in range(len(path_indices)-1))

        # Print results
        print(f"\nQAOA Results:")
        print(f"Optimal path: {' -> '.join(path)}")
        print(f"Total distance: {total_distance}")

        if visualize:
            # Show solution graph
            print("\nVisualizing Optimal Path:")
            visualize_graph(distances, cities, path)

        return path, total_distance

    except Exception as e:
        print(f"Error in QAOA execution: {e}")
        # Return a simple sequential path as fallback
        path = list(range(n_cities)) + [0]
        path = [cities[i] for i in path]
        return path, float('inf')

def visualize_graph(distances, cities, path=None):
    """
    Visualize TSP graph with path highlighting.
    """
    plt.figure(figsize=(10, 8))

    G = nx.Graph()
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            G.add_edge(cities[i], cities[j], weight=distances[i][j])

    pos = nx.circular_layout(G)

    # Draw all edges in light gray
    nx.draw_networkx_edges(G, pos,
                         edge_color='lightgray',
                         width=1,
                         style='dashed')

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos,
                               edge_labels=edge_labels,
                               font_size=10)

    if path is not None:
        # Draw solution path with arrows
        path_edges = list(zip(path[:-1], path))
        for start, end in path_edges:
            start_pos = pos[start]
            end_pos = pos[end]

            plt.arrow(start_pos[0], start_pos[1],
                     end_pos[0] - start_pos[0],
                     end_pos[1] - start_pos[1],
                     head_width=0.03,
                     head_length=0.05,
                     fc='red',
                     ec='red',
                     length_includes_head=True,
                     width=0.002)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                         node_color='lightblue',
                         node_size=1000,
                         edgecolors='black',
                         linewidths=2)

    nx.draw_networkx_labels(G, pos,
                          font_size=14,
                          font_weight='bold')

    if path is None:
        plt.title("Initial TSP Graph")
    else:
        total_distance = sum(distances[cities.index(path[i])][cities.index(path[i+1])]
                           for i in range(len(path)-1))
        plt.title(f"TSP Solution Path\nPath: {' → '.join(path)}\nTotal Distance: {total_distance}")

    plt.axis('off')

    # Add legend
    if path is not None:
        legend_elements = [
            plt.Line2D([0], [0], color='lightgray', linestyle='--',
                      label='Available Paths'),
            plt.Line2D([0], [0], color='red', label='Solution Path')
        ]
        plt.legend(handles=legend_elements, loc='lower center',
                  bbox_to_anchor=(0.5, -0.15))

    plt.tight_layout()
    plt.show()
