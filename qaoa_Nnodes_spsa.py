###############################################################
# Suppress all deprecation warnings
# Remove once we upgrade to qiskit > 1
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
###############################################################
# 
import sys
import time
import argparse
import threading
import numpy as np
from datetime import datetime

from itertools import combinations
from qiskit import Aer
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA, SPSA
from qiskit.utils import QuantumInstance
from qiskit.opflow import PauliOp
from qiskit.quantum_info import Pauli
from graph import visualize_graph


def decode_solution(result, n_cities):
    """
    Decode the QAOA result into a valid Traveling Salesperson Problem (TSP) path.

    This function extracts the most probable bitstring representation of the solution
    from the QAOA result's `eigenstate`, which may be a dictionary (e.g., from COBYLA)
    or an ndarray (e.g., from SPSA). The bitstring is converted into a state matrix
    representing the cities visited at different time steps, and a valid TSP path is 
    constructed.

    Args:
        result: The result object from the QAOA optimizer. It should have an `eigenstate`
                attribute, which may be a dictionary (for COBYLA) or a numpy ndarray (for SPSA).
        n_cities (int): The number of cities in the TSP problem.

    Returns:
        list: A list of city indices representing the order in which cities are visited,
              starting and ending at the same city to form a complete cycle.

    Handling of `eigenstate`:
        - **Dictionary (e.g., COBYLA)**: The function extracts the bitstring with the highest
          probability from the dictionary of bitstring counts.
        - **Numpy ndarray (e.g., SPSA)**: The function finds the index with the highest amplitude
          and converts it to a bitstring of appropriate length.
        - **Fallback**: If `eigenstate` is not present or its type is unrecognized, the function 
          returns a simple sequential path as a fallback.

    Notes:
        - The function constructs the path by checking which city is visited at each time step,
          ensuring no city is revisited until all cities have been visited.
        - If the `eigenstate` is complex (e.g., in ndarrays from SPSA), the function considers
          the magnitudes for identifying the most probable state.

    Example:
        Given an `eigenstate` of {'110001...': 0.45, '101010...': 0.3} (for COBYLA),
        or an ndarray of amplitudes (for SPSA), the function decodes the most probable 
        state into a TSP path representation.

    Raises:
        Exception: If there is an issue decoding the solution or parsing the `eigenstate`,
                   the function prints an error and returns a fallback path.
    """
    try:
        # Get the state with the highest probability from the counts dictionary or ndarray
        if hasattr(result, 'eigenstate'):
            eigenstate = result.eigenstate
            
            if isinstance(eigenstate, dict):
                # Handle dictionary of counts (e.g., COBYLA)
                counts = eigenstate
                max_bitstring = max(counts.items(), key=lambda x: x[1])[0]
                binary = format(int(max_bitstring, 2), f'0{n_cities * n_cities}b')
            
            elif isinstance(eigenstate, np.ndarray):
                # Handle ndarray (e.g., SPSA)
                # Find the index with the maximum amplitude
                max_index = np.argmax(np.abs(eigenstate))
                binary = format(max_index, f'0{n_cities * n_cities}b')
            
            else:
                # Fallback for unrecognized types
                print("Warning: Could not decode quantum state, using fallback path")
                path = list(range(n_cities)) + [0]
                return path

            # Convert to state matrix
            state_matrix = np.array(list(map(int, binary))).reshape(n_cities, n_cities)
        else:
            # Fallback: create a simple sequential path if no eigenstate is present
            print("Warning: No eigenstate found, using fallback path")
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


def create_tsp_graph(N):
    """
    Create a TSP problem with N cities labeled from A, B, C, ..., up to the Nth letter.
    Returns a symmetric distance matrix with zero diagonals.
    """
    # Generate random distance matrix
    distances = np.random.randint(1, 10, size=(N, N))  # Random distances between 1 and 9

    # Make the distance matrix symmetric
    distances = (distances + distances.T) / 2
    np.fill_diagonal(distances, 0)  # Set diagonal to 0, no self-loops

    # Generate city labels as uppercase letters
    cities = [chr(65 + i) for i in range(N)]  # A, B, C, ..., etc.

    return distances, cities


def create_cost_hamiltonian(distances):
    """
    Create the cost Hamiltonian for the Quantum Approximate Optimization Algorithm (QAOA) 
    for solving the Traveling Salesperson Problem (TSP).

    This function constructs a cost Hamiltonian that encodes the TSP problem in terms 
    of quantum operations. The cost Hamiltonian includes terms representing the distances 
    between cities and penalty terms to enforce problem constraints, ensuring that each city 
    is visited only once at each time step and each city is visited exactly once.

    Args:
        distances (numpy.ndarray): A 2D array representing the distance matrix where 
                                   distances[i][j] gives the distance between city i and city j.

    Returns:
        PauliSumOp: The total cost Hamiltonian as a sum of Pauli operators, which can be used 
                    in a QAOA algorithm to minimize the objective function.

    Components:
        - **Distance Terms**: Terms representing the distance between cities as part of the 
          cost function, weighted by the distance value.
        - **Penalty Terms**:
            - **One City per Time Step**: Penalties to ensure that no more than one city 
              is visited at each time step.
            - **Each City Visited Once**: Penalties to ensure that each city is visited 
              only once throughout the tour.

    Notes:
        - The Hamiltonian is built using Pauli Z operators, where the binary representation 
          of the problem is mapped to quantum qubits.
        - A strong penalty (defaulted to 20.0) is added to the Hamiltonian to enforce the 
          constraints strictly.

    Example:
        If `distances` is a 4x4 matrix representing distances between 4 cities, this function 
        creates a cost Hamiltonian involving 16 qubits (4 cities * 4 time steps).

    Raises:
        ValueError: If the distance matrix is not square or if other input requirements are not met.

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


last_print_time = 0
callback_interval = 0
def print_status(interval=60):
    """
    Function to print status with a timestamp every `interval` seconds.
    """
    global last_print_time
    global callback_interval

    while True:
        # Get the current time in seconds since the epoch
        current_time = time.time()
        
        # Print only if the specified interval has passed since the last print
        if current_time - last_print_time >= callback_interval:
            # Get the current timestamp
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Print the status message with the timestamp
            print(f"{current_time} - Simulation is still running...")
            # last_print_time = time.time() 

        # Wait for the specified interval before printing again
        time.sleep(callback_interval + 0.1)


def optimizer_callback(evaluation, params, value, step_size=None, accepted=None):
    """
    General callback function for optimizers in Qiskit (e.g., SPSA, COBYLA).
    Prints progress updates at a specified time interval (e.g., once every 1 minute).

    Args:
        evaluation (int): The current iteration number.
        params (numpy.ndarray): The current parameters being evaluated. This argument
            is kept for compatibility but is not printed.
        value (float): The current value of the objective function.
        step_size (float, optional): The step size used for the iteration (only applicable for SPSA).
        accepted (bool, optional): Whether the step was accepted (only applicable for SPSA).
        interval (int, optional): The minimum time interval (in seconds) between print statements.
            Default is 60 seconds.

    Prints:
        A formatted string that includes the current timestamp, iteration number,
        objective function value, step size (if applicable), and accepted status (if applicable).
    """
    global last_print_time
    global callback_interval
    
    # Get the current time in seconds since the epoch
    current_time = time.time()
    
    # Print only if the specified interval has passed since the last print
    if current_time - last_print_time >= callback_interval:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
        
        # Base message with evaluation and value
        message = f"{timestamp} - Iteration: {evaluation}, Value: {value:.5f}"
        
        # Add step size and accepted status if provided (SPSA-specific)
        if step_size is not None and accepted is not None:
            message += f", Step Size: {step_size:.5f}, Accepted: {accepted}"
        
        # Print the message
        print(message)
        
        # Update the last print time
        last_print_time = current_time


def solve_tsp_with_qaoa(distances):
    """
    Solve TSP using QAOA with SPSA and parallel experiements for faster execution.
    """
    global callback_interval
    callback_interval = 60

    # Create Hamiltonian
    cost_hamiltonian = create_cost_hamiltonian(distances)
    
    # Set up QAOA
    p = 3  # Number of QAOA layers
    optimizer = SPSA(maxiter=200, callback=optimizer_callback)

    backend = Aer.get_backend('qasm_simulator')
    quantum_instance = QuantumInstance(
        backend,
        shots=2048,
        seed_simulator=123,
        seed_transpiler=123,
        optimization_level=2,  # Optimize circuit depth - {0,1,2,3}
        backend_options={
            "max_parallel_threads": 16,  # Use up to 16 CPU cores for a single experiment
            "max_parallel_experiments": 8  # Run up to 8 experiments in parallel
        }
    )
    
    qaoa = QAOA(
        optimizer=optimizer,
        quantum_instance=quantum_instance,
        reps=p,
        initial_point=[np.pi/3] * (2*p)
    )
    
    # Run QAOA
    try:
        # Start the status printing in a separate thread
        print(f"\nSimulation starting!")
        start_time = time.time()
        status_thread = threading.Thread(target=print_status, args=(callback_interval,), daemon=True)
        status_thread.start()
        result = qaoa.compute_minimum_eigenvalue(cost_hamiltonian)
        end_time = time.time()
        runtime_duration = end_time - start_time
        minutes = runtime_duration // 60
        seconds = runtime_duration % 60
        print(f"Simulation ran for: {minutes:.0f} minutes and {seconds:.2f} seconds")
        print("Simulation complete!")

        return result, 0
        
    except Exception as e:
        print(f"Error in QAOA execution: {e}")
        return None, 1


def debug_result_info(result):
    """
    Print debug information about the QAOA result.
    """
    print(f"\nQAOA Debug Result:")
    print(f"Result type: {type(result)}")
    # print(f"Available attributes: {dir(result)}")
    
    if hasattr(result, 'eigenstate'):
        eigenstate = result.eigenstate
        print(f"Eigenstate type: {type(eigenstate)}")
        
        if isinstance(eigenstate, dict):
            max_state = max(eigenstate, key=eigenstate.get)
            print(f"Most probable state (dict): {max_state} with probability {eigenstate[max_state]}")
        elif isinstance(eigenstate, np.ndarray):
            print(f"Eigenstate (ndarray): {eigenstate[:5]}... (truncated for display)")
        else:
            print("Warning: Eigenstate format not recognized.")


def decode_and_print_solution(result, n_cities, distances, cities):
    """
    Decode the solution, calculate the total distance, and print the results,
    including the distance between each step in the path.
    """
    try:
        path_indices = decode_solution(result, n_cities)
        path = [cities[i] for i in path_indices]

        # Calculate total distance and print distances between each step
        total_distance = 0
        print(f"\nOptimal path: {' -> '.join(path)}")
        print("Distances between steps:")
        
        for i in range(len(path_indices) - 1):
            start = path_indices[i]
            end = path_indices[i + 1]
            step_distance = distances[start][end]
            total_distance += step_distance
            print(f"{cities[start]} -> {cities[end]}: {step_distance:.1f}")
        
        # Print the total distance
        print(f"\nTotal distance: {total_distance:.1f}")

        return path

    except Exception as e:
        print(f"Error decoding solution: {e}")
        sys.exit(1)


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Solve TSP using QAOA and optionally save the graph.")
    parser.add_argument('--nodes', type=int, required=True, help='The number of cities (an integer)')
    parser.add_argument('--save-graph', action='store_true', help='Save the graph to a PNG file instead of displaying it')
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Extract the number of nodes (cities)
    try:
        N = int(args.nodes)
        if N < 2:
            raise ValueError("N must be 2 or greater.")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Check if the --save-graph flag is set
    save_graph = args.save_graph

    # Create problem instance
    distances, cities = create_tsp_graph(N)
    print(f"\nCities:\n{cities}\n\nDistances:")
    print(np.array2string(distances, formatter={'float_kind': lambda x: f"{x:.1f}".rstrip('.')}))
    
    n_cities = len(distances)
    # Solving the TSP with QAOA
    result, err = solve_tsp_with_qaoa(distances)
    if err:
        sys.exit(1)

    # Print debug information
    debug_result_info(result)

    # Decode and print the solution
    path = decode_and_print_solution(result, n_cities, distances, cities)

    # Visualize the solution
    print("\nOptimal Path Visualization:")
    #visualize_graph(distances, cities, path, save_graph)
    sys.exit(0)


if __name__ == "__main__":
    main()
