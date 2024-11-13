import argparse
import json
import os
import sys
import time
import traceback
import numpy as np
import penalty
from bruteforce import solve_tsp_with_bruteforce
from graph import visualize_graph
from qaoa_qiskit1_2 import solve_tsp_with_qaoa

class Algo:
    QAOA=1
    BRUTEFORCE=2
    BOTH=3

FIXED_TSP = {
    # TSP problem with 3 cities
    "3": {
        "distances": np.array([
            [0, 2, 4],  # V1 to [V1, V2, V3]
            [2, 0, 1],  # V2 to [V1, V2, V3]
            [4, 1, 0],  # V3 to [V1, V2, V3]
        ]),
        "cities": ['V1', 'V2', 'V3', 'V4'],
        "minCost": 7.0
    },

    # TSP problem with 4 cities
    "4": {
        "distances": np.array([
            [0, 2, 4, 1],  # V1 to [V1, V2, V3, V4]
            [2, 0, 1, 3],  # V2 to [V1, V2, V3, V4]
            [4, 1, 0, 5],  # V3 to [V1, V2, V3, V4]
            [1, 3, 5, 0]   # V4 to [V1, V2, V3, V4]
        ]),
        "cities": ['V1', 'V2', 'V3', 'V4'],
        "minCost": 9.0
    },

    # TSP problem with 5 cities
    "5": {
        "distances": np.array([
            [0, 2, 4, 1, 1],  # V1 to [V1, V2, V3, V4, V5]
            [2, 0, 1, 3, 1],  # V2 to [V1, V2, V3, V4, V5]
            [4, 1, 0, 5, 2],  # V3 to [V1, V2, V3, V4, V5]
            [1, 3, 5, 0, 3],  # V4 to [V1, V2, V3, V4, V5]
            [1, 1, 2, 3, 0],  # V5 to [V1, V2, V3, V4, V5]
        ]),
        "cities": ['V1', 'V2', 'V3', 'V4', 'V5'],
        "minCost": 8.0
    },

    # TSP problem with 6 cities
    "6": {
        "distances": np.array([
            [0.0, 4.0, 6.0, 9.0, 5.5, 5.0], # V1 to [V1, V2, V3, V4, V5, V6]
            [4.0, 0.0, 4.0, 2.0, 2.5, 6.0], # V2 to [V1, V2, V3, V4, V5, V6]
            [6.0, 4.0, 0.0, 6.5, 7.5, 4.5], # V3 to [V1, V2, V3, V4, V5, V6]
            [9.0, 2.0, 6.5, 0.0, 6.5, 4.5], # V4 to [V1, V2, V3, V4, V5, V6]
            [5.5, 2.5, 7.5, 6.5, 0.0, 7.5], # V5 to [V1, V2, V3, V4, V5, V6]
            [5.0, 6.0, 4.5, 4.5, 7.5, 0.0], # V6 to [V1, V2, V3, V4, V5, V6]
        ]),
        "cities": ['V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
        "minCost": 25.0
    }
}

def create_tsp_graph(n):
    """
    Create a TSP problem with n cities labeled V1, V2, .., Vn.
    """
    n_str = str(n)
    print(n_str, FIXED_TSP.keys())
    # Use fixed problem definitions for up to 6 nodes
    if n_str in FIXED_TSP.keys():
        tsp = FIXED_TSP[n_str]
        return tsp["distances"], tsp["cities"], tsp["minCost"]

    distance_matrix, city_names, min_cost = load_tsp(n)
    if distance_matrix is None or city_names is None or min_cost is None:
        if distance_matrix is None:
            print(f"No saved problem for N={n}. Generating TSP for {n} nodes")
            # Generate random distance matrix
            distance_matrix = np.random.randint(1, 10, size=(n, n))
            # Make the matrix symmetric and set diagonal to zero
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)
        if city_names is None:
            # Generate list of city names
            city_names = [f"V{i+1}" for i in range(n)]
        if min_cost is None:
            min_cost = run_bruteforce(distance_matrix, city_names, save_graph=False)
        # Save the TSP problem in a file under ./problems directory
        save_tsp(n, distance_matrix, city_names, min_cost)

    return distance_matrix, city_names, min_cost

def run_bruteforce(
        distances, cities,
        save_graph=True):
    """
    Runs the experiment for a given problem using Bruteforce
    """
    print("\nUsing Bruteforce to solve..")
    start_time = time.time()
    result_path, shortest_distance = solve_tsp_with_bruteforce(distances, cities)
    end_time = time.time()

    runtime_duration = end_time - start_time
    minutes = runtime_duration // 60
    seconds = runtime_duration % 60
    print(f"Bruteforce completed in: {minutes:.0f} minutes {seconds:.2f} seconds")

    if result_path and save_graph:
        # Generate solution graph
        print("\nVisualizing Optimal Path computed by Bruteforce:")
        try:
            visualize_graph(distances, cities, result_path, save=True, save_prefix='Brute_')
        except Exception as ex:
            print(f"Error in visualize_graph for Bruteforce: {ex}")
            traceback.print_exc()
    
    return print_path(distances, cities, result_path)


def run_qaoa(
        distances, cities,
        optimizer_choice:str,
        optimizer_maxiter:int=3,
        use_simulator=True,
        save_graph=True):
    """
    Runs the experiment for a given problem using QAOA
    """
    print("\nUsing QAOA..")
    start_time = time.time()

    result_path, err = solve_tsp_with_qaoa(distances, cities,
        optimizer_choice=optimizer_choice,
        optimizer_maxiter=optimizer_maxiter,
        use_simulator=use_simulator
        )
    if err:
        print(f"Error running QAOA: {err}")
        exit(err)

    end_time = time.time()
    runtime_duration = end_time - start_time
    minutes = runtime_duration // 60
    seconds = runtime_duration % 60
    print(f"QAOA completed in: {minutes:.0f} minutes {seconds:.2f} seconds")

    print("\nQAOA Results:")
    total_distance = print_path(distances, cities, result_path)

    run_mode = 'SIM' if use_simulator else 'QPU'
    save_prefix = f'{run_mode}_{optimizer_choice}-{optimizer_maxiter}_'

    if save_graph:
        # Generate solution graph
        print("\nVisualizing Optimal Path computed by QAOA:")
        try:
            visualize_graph(distances, cities, result_path, save=True, save_prefix='qaoa_'+save_prefix)
        except Exception as ex:
            print(f"Error in visualize_graph for QAOA: {ex}")
            traceback.print_exc()
    
    return total_distance

approxRatios = {}
def run_experiment(
        num_of_nodes:int,
        optimizer_choice:str,
        optimizer_maxiter:int=3,
        use_simulator=True,
        save_graph=True, algorithms=3):
    
    global approxRatios
    if(algorithms not in [Algo.QAOA, Algo.BRUTEFORCE, Algo.BOTH]):
        print("Invalid value passed for algorithms")

    print("\n============================================\n")
    print(f"Running Experiment in {'Simulator' if use_simulator else 'IBM Cloud'} with:")
    print(f"Cities={num_of_nodes}, Optimizer={optimizer_choice}, Maxiter={optimizer_maxiter}")

    # Create problem definition
    distances, cities, min_cost = create_tsp_graph(num_of_nodes)

    print(f"Solving TSP for {len(cities)} cities:\n", cities, "\nDistance Matrix:\n", distances)

    if(algorithms & Algo.BRUTEFORCE):
        run_bruteforce(distances, cities, save_graph)

    qaoa_distance = None
    if(algorithms & Algo.QAOA):
        qaoa_distance = run_qaoa(distances, cities, optimizer_choice, optimizer_maxiter, use_simulator, save_graph)
        ar = qaoa_distance/min_cost
        configKey = f"N{num_of_nodes}:{optimizer_choice[0]}:I{optimizer_maxiter}:{penalty.key()}"
        if configKey in approxRatios:
            approxRatios[configKey].append(ar)
        else:
            approxRatios[configKey] = [ar]


def print_path(distances, cities, path):
        path_indices = [ cities.index(c) for c in path]

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
        
        print(f"\nUsing Penalty Coefficient: MaxDistance {penalty.op} {penalty.const}")
        # Print the total distance
        print(f"\nTotal distance: {total_distance:.1f}")
        return total_distance



# Function to save array to a file
def save_tsp(n: int, distances: np.ndarray, cities: list, min_cost: float):
    # Ensure the directory "problems" exists
    os.makedirs("problems", exist_ok=True)
    filepath = os.path.join("problems", f"{n}.tsp")
    with open(filepath, 'w') as file:
        tsp = {
            "distances": distances.tolist(),
            "cities": cities,
            "minCost": min_cost
        }
        json.dump(tsp, file)

# Function to load array from a file
def load_tsp(n: int) -> np.ndarray:
    filepath = os.path.join("problems", f"{n}.tsp")
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            tsp = json.load(file)
            if(isinstance(tsp, np.ndarray)):
                return  np.array(tsp), None, None
            elif isinstance(tsp, dict):
                return np.array(tsp["distances"]), tsp["cities"], tsp["minCost"]
            else:
                return None, None, None

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Solve TSP using QAOA and optionally save the graph.")

    # Define the --nodes argument as an integer
    parser.add_argument("--nodes",
        type=int,
        help="The number of cities(an integer > 2)",
        default=3)

    # Define the --save-graph flag as a boolean
    parser.add_argument("--save-graph",
        action="store_true",
        help="Save the graph to a PNG file instead of displaying it",
        default=True)

    # Define the --real flag as a boolean
    parser.add_argument("--real",
        action="store_true",
        help="Run on IBM's real QPU",
        default=False)
    
    # Define the --mode flag as a boolean
    parser.add_argument("--algo",
        type=int,
        choices=[Algo.QAOA, Algo.BRUTEFORCE, Algo.BOTH],
        help="Algorithms to run (1=>QAOA only, 2=>Bruteforce only, 3=>Both)",
        default=Algo.BOTH)
    
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--spsa', action='store_true', help="Use the SPSA optimizer", default=True)
    group.add_argument('--cobyla', action='store_true', help="Use the COBYLA optimizer", default=False)

    # Define the --real flag as a boolean
    parser.add_argument("--maxiter",
        type=int,
        help="Value for maxiter parameter passed to optimizers(an integer). This affects number of workloads created if running on IBM platform",
        default=3)
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Set variables based on the command line arguments
    num_of_nodes = args.nodes
    save_graph = args.save_graph
    use_simulator = not args.real
    optimizer_maxiter = args.maxiter
    algo = args.algo

    # Extract the number of nodes (cities)
    try:
        num_of_nodes = int(args.nodes)
        if num_of_nodes < 2:
            raise ValueError("Number of nodes must be 2 or greater.")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Determine which optimizer to use based on the parsed arguments
    if args.cobyla:
        optimizer_choice = 'COBYLA'
    else:
        optimizer_choice = 'SPSA'

    print(f"Save Graph: {save_graph}")
    print(f"Use Simulator: {use_simulator}")
    print(f"Optimizer: {optimizer_choice}")

    run_experiment(num_of_nodes, optimizer_choice, optimizer_maxiter, use_simulator=use_simulator, save_graph=save_graph, algorithms=algo)