import sys
from itertools import combinations
import numpy as np
from qiskit.result.distributions.quasi import QuasiDistribution
from qiskit_aer import QasmSimulator
from qiskit_algorithms.minimum_eigensolvers.qaoa import QAOA
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.primitives import BackendSampler
import time
import traceback

def debug_result_info(result, verbose=False):
    """
    Print debug information about the QAOA result.
    """
    print("\nQAOA Debug Result:")
    if verbose:
        print(f"Result type: {type(result)}")
    
    if hasattr(result, 'eigenstate'):
        eigenstate = result.eigenstate
        if verbose:
            print(f"Eigenstate type: {type(eigenstate)}")
        
        if isinstance(eigenstate, QuasiDistribution):
            print(f"Eigenstate (QuasiDistribution):")
            items = list(eigenstate.binary_probabilities().items())
            print(items if verbose else f"{items[:5]} ... (truncated for display)")
        elif isinstance(eigenstate, dict):
            print(f"Eigenstate (dict):")
            items = list(eigenstate.items())
            print(items if verbose else f"{items[:5]} ... (truncated for display)")
        elif isinstance(eigenstate, np.ndarray):
            print("Eigenstate (ndarray):")
            print(eigenstate if verbose else f"{eigenstate[:5]} ... (truncated for display)")
        else:
            print("Warning: Eigenstate format not recognized.")

    print("\n")

def decode_solution(result, n_cities):
    """
    Decode the QAOA result into a valid TSP path.
    Now properly handles dictionary output from QAOA.
    """
    # Get the state with highest probability from the counts dictionary
    if not hasattr(result, 'eigenstate'):
        raise Exception("Cannot Decode: The 'eigenstate' attribute is missing in 'result'")

    eigenstate = result.eigenstate

    if isinstance(eigenstate, QuasiDistribution):
        # Handle QuasiDistribution (e.g., from Qiskit 1.2)
        binaryKeyedDict = eigenstate.binary_probabilities()
        max_probability_key = max(binaryKeyedDict.items(), key=lambda x: x[1])[0]
        max_probability = binaryKeyedDict.get(max_probability_key)
        binary = max_probability_key
    elif isinstance(eigenstate, dict):
        # Handle dictionary of counts (e.g., COBYLA)
        items = eigenstate.items()
        max_probability_key = max(items, key=lambda x: x[1])[0]
        max_probability = eigenstate.get(max_probability_key)
        binary = format(int(max_probability_key, 2), f'0{n_cities * n_cities}b')
    
    elif isinstance(eigenstate, np.ndarray):
        # Handle ndarray (e.g., SPSA)
        # Find the index with the maximum amplitude
        max_probability_key = np.argmax(np.abs(eigenstate))
        max_probability = eigenstate[max_probability_key]
        binary = format(max_probability_key, f'0{n_cities * n_cities}b')
    
    else:
        # Error for unrecognized types
        raise Exception(f"Could not decode quantum state from result type: {type(eigenstate)}")

    print("MAX PROBABILITY STATE: ", binary)
    print("MAX PROBABILITY VALUE: ", max_probability)

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

def decode_and_print_solution(result, distances, cities):
    """
    Decode the solution, calculate the total distance, and print the results,
    including the distance between each step in the path.
    """
    try:
        path_indices = decode_solution(result, len(cities))
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

    except Exception as ex:
        print(f"Error decoding solution: {ex}")
        traceback.print_exc()
        return None


last_print_time = time.time()
callback_interval = 30

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
                    pauliList.append((pauli, distances[i][j]))

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

def solve_tsp_with_qaoa(distances, cities, 
                        optimizer_choice:str='spsa',
                        optimizer_maxiter:int=10,
                        use_simulator:bool=True):
    """
    Solve TSP using QAOA on IBM Quantum hardware.
    """
    # Create Hamiltonian
    cost_hamiltonian = create_cost_hamiltonian(distances)

    # Set up QAOA
    p = 3  # Number of QAOA layers (keep small for hardware constraints)

    if optimizer_choice == 'cobyla':
        optimizer = COBYLA(maxiter=optimizer_maxiter, callback=cobyla_callback)
    else:
        optimizer = SPSA(maxiter=optimizer_maxiter, callback=spsa_callback)

    if use_simulator:
        backend = QasmSimulator()
    else:
        # Save the IBM Quantum Experience Credentials only for first run and DO NOT COMMIT the Token in GIT repo!
        # QiskitRuntimeService.save_account(channel="ibm_quantum", token="<YOUR-TOKEN>", overwrite=True, set_as_default=True)
        
        # Load IBMQ account and select backend
        service = QiskitRuntimeService(channel="ibm_quantum")
        # Get the least busy backend with enough qubits
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=50)
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
    print(f"Running {p} layer QAOA optimizer={type(optimizer).__name__} maxiter={optimizer_maxiter}")
    try:
        result = qaoa.compute_minimum_eigenvalue(cost_hamiltonian)

        # Print details of result to help debugging issues
        debug_result_info(result)

        print("QAOA Results:")
        # Decode and print the solution
        path = decode_and_print_solution(result, distances, cities)

        return path, 0

    except Exception as e:
        print(f"Error in QAOA execution: {e}")
        traceback.print_exc()
        return None, 1
