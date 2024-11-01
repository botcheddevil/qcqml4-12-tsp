import warnings
import sys
import time
import argparse
import threading
import numpy as np
from datetime import datetime

from itertools import combinations
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit.primitives import Sampler
from qiskit.quantum_info import SparsePauliOp
from graph import visualize_graph

def decode_solution(result, n_cities):
    """
    Decode the QAOA result into a valid Traveling Salesperson Problem (TSP) path.

    This function extracts the most probable bitstring representation of the solution
    from the `eigenstate` in the result. The `eigenstate` can be either a dictionary 
    (e.g., from COBYLA) or a QuasiDistribution (e.g., from Qiskit 1.2 and later).
    The function decodes the bitstring to form a valid TSP path.

    Args:
        result: The result object from the QAOA optimizer. It should have an `eigenstate`
                attribute, which may be a dictionary or QuasiDistribution.
        n_cities (int): The number of cities in the TSP problem.

    Returns:
        list: A list of city indices representing the order in which cities are visited,
              starting and ending at the same city to form a complete cycle.

    Handling of `eigenstate`:
        - **Dictionary**: Extracts the bitstring with the highest probability.
        - **QuasiDistribution**: Extracts the bitstring with the highest probability or amplitude.
        - **Fallback**: If no valid `eigenstate` is found, returns a sequential fallback path.

    Notes:
        - The function constructs the path by checking which city is visited at each time step,
          ensuring no city is revisited until all cities have been visited.
        - Handles both real and complex eigenstates by considering magnitudes.

    Raises:
        ValueError: If an unrecognized type of `eigenstate` is encountered.
    """
    try:
        # Extract the most probable bitstring from the eigenstate
        if hasattr(result, 'eigenstate'):
            eigenstate = result.eigenstate
            
            if isinstance(eigenstate, dict):
                # Handle dictionary (e.g., COBYLA)
                max_bitstring = max(eigenstate.items(), key=lambda x: x[1])[0]
            elif isinstance(eigenstate, QuasiDistribution):
                # Handle QuasiDistribution (e.g., from Qiskit 1.2)
                max_bitstring = max(eigenstate.items(), key=lambda x: x[1])[0]
            else:
                raise ValueError("Unrecognized eigenstate type. Using fallback path.")
        
        else:
            print("Warning: No eigenstate found, using fallback path")
            return list(range(n_cities)) + [0]

        # Convert max_bitstring to binary string if it's an integer
        if isinstance(max_bitstring, int):
            binary = format(max_bitstring, f'0{n_cities * n_cities}b')
        elif isinstance(max_bitstring, str):
            binary = max_bitstring
        else:
            raise ValueError("Unexpected type for max_bitstring")

        # Convert to state matrix
        state_matrix = np.array(list(map(int, binary))).reshape(n_cities, n_cities)

        # Build the valid path
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
        print(f"Error in solution decoding: {e}")
        print("Using fallback path")
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
                    pauli_op = SparsePauliOp.from_list([(''.join(pauli_str), distances[i][j] / 4)])
                    
                    cost_ops.append(pauli_op)
    
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
            pauli_op = SparsePauliOp.from_list([(''.join(pauli_str), penalty)])
            
            cost_ops.append(pauli_op)
    
    # Each city visited once
    for i in range(n_cities):
        for t1, t2 in combinations(range(n_cities), 2):
            qubit1 = i * n_cities + t1
            qubit2 = i * n_cities + t2
            
            pauli_str = ['I'] * n_qubits
            pauli_str[qubit1] = 'Z'
            pauli_str[qubit2] = 'Z'
            pauli_op = SparsePauliOp.from_list([(''.join(pauli_str), penalty)])
            
            cost_ops.append(pauli_op)
    
    return sum(cost_ops)


last_print_time = 0
callback_interval = 0


def print_status(callback_interval=0):
    """
    Function to print status with a timestamp every `interval` seconds.
    """
    global last_print_time

    while True:
        # Get the current time in seconds since the epoch
        current_time = time.time()
        
        # Print only if the specified interval has passed since the last print
        if current_time - last_print_time >= callback_interval + 60:
            # Get the current timestamp
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Print the status message with the timestamp
            print(f"{current_time} - Simulation is still running...")
            # last_print_time = time.time() 

        # Wait for the specified interval before printing again
        time.sleep(callback_interval)


def cobyla_callback(evaluation, *args):
    """
    Callback function for the COBYLA optimizer in Qiskit.
    Prints progress updates at a specified time interval (e.g., once every 30 seconds).

    Args:
        evaluation (int): The current iteration number.

    Prints:
        A formatted string that includes the current timestamp and iteration number.
    """
    global last_print_time
    global callback_interval

    # Get the current time in seconds since the epoch
    current_time = time.time()
    
    # Print only if the specified interval has passed since the last print
    if current_time - last_print_time >= callback_interval:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
        message = f"{timestamp} - Iteration {evaluation}"
        
        # Print the message
        print(message)
        
        # Update the last print time
        last_print_time = current_time


def spsa_callback(evaluation, params, value, step_size=None, accepted=None):
    """
    Callback function for the SPSA optimizer in Qiskit.
    Prints progress updates at a specified time interval (e.g., once every 30 seconds).

    Args:
        evaluation (int): The current iteration number.
        params (numpy.ndarray): The current parameters being evaluated.
        value (float): The current value of the objective function.
        step_size (float, optional): The step size used for the iteration.
        accepted (bool, optional): Whether the step was accepted.

    Prints:
        A formatted string that includes the current timestamp, iteration number,
        objective function value, step size, and accepted status.
    """
    global last_print_time
    global callback_interval

    # Get the current time in seconds since the epoch
    current_time = time.time()
    
    current_time = time.time()
    if current_time - last_print_time >= callback_interval:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
        message = f"{timestamp} - Iteration {evaluation}: Value {value:.5f}, Step Size {step_size:.5f}, Accepted: {accepted}"
        print(message)
        last_print_time = current_time


def solve_tsp_with_qaoa(distances, optimizer_choice):
    """
    Solve TSP using QAOA with SPSA for optimization and StatevectorSampler for execution.
    """
    global callback_interval
    callback_interval = 180

    # Create cost Hamiltonian (assumes this function is defined)
    cost_hamiltonian = create_cost_hamiltonian(distances)
    
    # Set up optimizer and QAOA parameters
    reps = 2
    # Set up optimizer and QAOA parameters
    reps = 2
    if optimizer_choice == 'spsa':
        optimizer = SPSA(maxiter=200, callback=spsa_callback)
    elif optimizer_choice == 'cobyla':
        optimizer = COBYLA(maxiter=600, callback=cobyla_callback)
    else:
        print("Invalid optimizer choice. Use --spsa or --cobyla.")
        return None, 1
    
    # Create a StatevectorSampler for circuit execution
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    sampler = Sampler()

    # Create QAOA instance with StatevectorSampler for the optimization phase
    qaoa = QAOA(sampler, optimizer,reps=reps, initial_point=[np.pi / 3] * (2 * reps))
    
    try:
        # Run QAOA for optimization
        print(f"\nOptimization starting ({optimizer_choice})!")

        start_time = time.time()
        status_thread = threading.Thread(target=print_status, args=(callback_interval,), daemon=True)
        status_thread.start()
        result = qaoa.compute_minimum_eigenvalue(cost_hamiltonian)
        end_time = time.time()
        runtime_duration = end_time - start_time
        minutes = runtime_duration // 60
        seconds = runtime_duration % 60
        print(f"Optimization complete! Runtime: {minutes:.0f} minutes and {seconds:.2f} seconds")

        return result, 0
    
    except Exception as e:
        print(f"Error during QAOA execution: {e}")
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--spsa', action='store_true', help="Use the SPSA optimizer")
    group.add_argument('--cobyla', action='store_true', help="Use the COBYLA optimizer")
    
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

    # Determine which optimizer to use based on the parsed arguments
    if args.spsa:
        optimizer_choice = 'spsa'
    elif args.cobyla:
        optimizer_choice = 'cobyla'

    # Create problem instance
    distances, cities = create_tsp_graph(N)
    print(f"\nCities:\n{cities}\n\nDistances:")
    print(np.array2string(distances, formatter={'float_kind': lambda x: f"{x:.1f}".rstrip('.')}))
    
    n_cities = len(distances)
    # Solving the TSP with QAOA
    result, err = solve_tsp_with_qaoa(distances, optimizer_choice)
    if err:
        sys.exit(1)

    # Print debug information
    debug_result_info(result)

    # Decode and print the solution
    path = decode_and_print_solution(result, n_cities, distances, cities)

    # Visualize the solution
    print("\nOptimal Path Visualization:")
    visualize_graph(distances, cities, path, save_graph)
    sys.exit(0)


if __name__ == "__main__":
    main()
