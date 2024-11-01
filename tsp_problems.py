import argparse
import time
import numpy as np
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
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Set variables based on the command line arguments
    numOfNodes = args.nodes
    saveGraph = args.save_graph
    useSimulator = not args.real
    
    print(f"Save Graph: {saveGraph}")
    print(f"Use Simulator: {useSimulator}")

    for n in range(3,numOfNodes+1):
        start_time = time.time()
        match numOfNodes:
            case 3: distances, cities = create_tsp_graph_3nodes()
            case 4: distances, cities = create_tsp_graph_4nodes()
            case 5: distances, cities = create_tsp_graph_5nodes()
            case _: distances, cities = create_tsp_graph(numOfNodes)
        
        solve_tsp_with_qaoa(distances, cities, visualize=saveGraph, useSimulator=useSimulator, saveGraph=saveGraph)

        end_time = time.time()
        runtime_duration = end_time - start_time
        minutes = runtime_duration // 60
        seconds = runtime_duration % 60
        print(f"Script ran for: {minutes} minutes {seconds} seconds")
