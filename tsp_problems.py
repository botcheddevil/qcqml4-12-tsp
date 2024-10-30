import numpy as np
from qaoa_solution import solve_tsp_with_qaoa

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

if __name__ == "__main__":
    optimal_path, total_distance = solve_tsp_with_qaoa(create_tsp_graph_3nodes)
    # optimal_path, total_distance = solve_tsp_with_qaoa(create_tsp_graph_4nodes)
    # optimal_path, total_distance = solve_tsp_with_qaoa(create_tsp_graph_5nodes)