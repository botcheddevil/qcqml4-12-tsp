import traceback
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from graph import visualize_graph
from itertools import combinations
from qiskit_aer import QasmSimulator
from qiskit_algorithms.minimum_eigensolvers.qaoa import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.primitives import BackendSampler

def decode_solution(result, n_cities):
    """
    Decode the QAOA result into a valid TSP path.
    Now properly handles dictionary output from QAOA.
    """
    try:
        
        if hasattr(result, 'eigenstate'):
            print(f"Eigenstate type: {type(result.eigenstate)}")

        # Get the state with highest probability from the counts dictionary
        if hasattr(result, 'eigenstate') and isinstance(result.eigenstate, dict):
            # Handle dictionary of counts
            counts = result.eigenstate
            max_probability_state = max(counts.items(), key=lambda x: x[1])[0]
            print("MAX PROBABILITY STATE: ", max_probability_state)
            print("MAX PROBABILITY VALUE: ", counts.get(max_probability_state))
            binary = format(max_probability_state, f'0{n_cities * n_cities}b')
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
        traceback.print_exc()
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
    pauliList = []
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
                    pauli = ''.join(pauli_str)
                    pauliList.append((pauli, distances[i][j] / 4))

    cost_ops.append(SparsePauliOp.from_list(pauliList))

    # Add sufficiently strong penalty terms
    penalty = 20.0

    # One city per time step
    pauliList = []
    for t in range(n_cities):
        for i, j in combinations(range(n_cities), 2):
            qubit1 = i * n_cities + t
            qubit2 = j * n_cities + t

            pauli_str = ['I'] * n_qubits
            pauli_str[qubit1] = 'Z'
            pauli_str[qubit2] = 'Z'
            pauli = ''.join(pauli_str)
            pauliList.append((pauli, penalty))

    cost_ops.append(SparsePauliOp.from_list(pauliList,))

    # Each city visited once
    pauliList = []
    for i in range(n_cities):
        for t1, t2 in combinations(range(n_cities), 2):
            qubit1 = i * n_cities + t1
            qubit2 = i * n_cities + t2

            pauli_str = ['I'] * n_qubits
            pauli_str[qubit1] = 'Z'
            pauli_str[qubit2] = 'Z'
            pauli = ''.join(pauli_str)
            pauliList.append((pauli, penalty))

    cost_ops.append(SparsePauliOp.from_list(pauliList))

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
        print("Visualizing Problem Graph:")
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

    sampler = BackendSampler(backend)
    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=p,
        initial_point=[np.pi/3] * (2*p)
    )

    # Run QAOA
    try:
        result = qaoa.compute_minimum_eigenvalue(cost_hamiltonian)

        # Debug information
        print("\nQAOA Result Details:")
        print(f"Result type: {type(result)}")

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
        traceback.print_stack()
        # Return a simple sequential path as fallback
        path = list(range(n_cities)) + [0]
        path = [cities[i] for i in path]
        return path, float('inf')
