import argparse
import sys
import time
import numpy as np
from graph import visualize_graph
from qaoa_qiskit1_2 import solve_tsp_with_qaoa

def create_tsp_graph_3nodes():
    """
    Create a TSP problem with 3 cities labeled A, B, C
    """
    distances = np.array([
        [0, 2, 4],  # A to [A, B, C]
        [2, 0, 1],  # B to [A, B, C]
        [4, 1, 0],  # C to [A, B, C]
    ])
    cities = ['A', 'B', 'C']
    return distances, cities    

def create_tsp_graph_4nodes():
    """
    Create a TSP problem with 4 cities labeled A, B, C, D.
    """
    distances = np.array([
        [0, 2, 4, 1],  # A to [A, B, C, D]
        [2, 0, 1, 3],  # B to [A, B, C, D]
        [4, 1, 0, 5],  # C to [A, B, C, D]
        [1, 3, 5, 0]   # D to [A, B, C, D]
    ])
    cities = ['A', 'B', 'C', 'D']
    return distances, cities

def create_tsp_graph_5nodes():
    """
    Create a TSP problem with 4 cities labeled A, B, C, D, E.
    """
    distances = np.array([
        [0, 2, 4, 1, 1],  # A to [A, B, C, D, E]
        [2, 0, 1, 3, 1],  # B to [A, B, C, D, E]
        [4, 1, 0, 5, 2],  # C to [A, B, C, D, E]
        [1, 3, 5, 0, 3],  # D to [A, B, C, D, E]
        [1, 1, 2, 3, 0],  # E to [A, B, C, D, E]
    ])
    cities = ['A', 'B', 'C', 'D', 'E']
    return distances, cities


def create_tsp_graph(n):
    """
    Create a TSP problem with n cities labeled V1, V2, .., Vn.
    """
    # Generate random distance matrix
    distance_matrix = np.random.randint(1, 100, size=(n, n))
    # Make the matrix symmetric and set diagonal to zero
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)

    # Generate list of city names
    city_names = [f"V{i+1}" for i in range(n)]

    return distance_matrix, city_names


def run_experiment(
        num_of_nodes:int,
        optimizer_choice:str,
        optimizer_maxiter:int=3,
        use_simulator=True,
        save_graph=False):
    print("\n============================================\n")
    print(f"Running Experiment with: Cities={num_of_nodes}, Optimizer={optimizer_choice}, Maxiter={optimizer_maxiter}")

    start_time = time.time()
    # Create problem definition
    match num_of_nodes:
        case 3: distances, cities = create_tsp_graph_3nodes()
        case 4: distances, cities = create_tsp_graph_4nodes()
        case 5: distances, cities = create_tsp_graph_5nodes()
        case _: distances, cities = create_tsp_graph(num_of_nodes)

    print(f"Solving TSP for {len(cities)} cities:\n", cities, "\nDistance Matrix:\n", distances)

    if save_graph:
        # Show initial graph
        print("Visualizing Problem Graph:")
        visualize_graph(distances, cities, save=True)

    optimalPath, err = solve_tsp_with_qaoa(distances, cities,
        optimizer_choice=optimizer_choice,
        optimizer_maxiter=optimizer_maxiter,
        use_simulator=use_simulator
        )
    
    if not err:
        if save_graph:
            # Show solution graph
            print("\nVisualizing Optimal Path:")
            visualize_graph(distances, cities, optimalPath, save=True)

    end_time = time.time()
    runtime_duration = end_time - start_time
    minutes = runtime_duration // 60
    seconds = runtime_duration % 60
    print(f"Script ran for: {minutes:.0f} minutes {seconds:.2f} seconds")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script to handle --sim flag and --n integer.")

    # Define the --nodes argument as an integer
    parser.add_argument("--nodes",
        type=int,
        help="The number of cities (an integer)",
        default=3)

    # Define the --save-graph flag as a boolean
    parser.add_argument(
        "--save-graph",
        action="store_true",
        help="Save the graph to a PNG file instead of displaying it",
        default=False)

    # Define the --real flag as a boolean
    parser.add_argument(
        "--real",
        action="store_true",
        help="Run on IBM's real QPU",
        default=False)
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--spsa', action='store_true', help="Use the SPSA optimizer", default=True)
    group.add_argument('--cobyla', action='store_true', help="Use the COBYLA optimizer")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Set variables based on the command line arguments
    num_of_nodes = args.nodes
    save_graph = args.save_graph
    use_simulator = not args.real

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

    run_experiment(3, optimizer_choice, 5, use_simulator, save_graph)