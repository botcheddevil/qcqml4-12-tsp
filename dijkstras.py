import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import permutations

from graph import visualize_graph
from tsp_problems import create_tsp_graph


def create_graph(distances, cities):
    """
    Create a NetworkX graph from the distance matrix.
    """
    n_cities = len(cities)
    G = nx.Graph()
    for i in range(n_cities):
        for j in range(i + 1, n_cities):
            G.add_edge(cities[i], cities[j], 
                        weight=distances[i][j])
    return G

def dijkstra_shortest_path(graph, start, end):
    """
    Find shortest path between two cities using Dijkstra's algorithm.
    
    Args:
        start (str): Starting city
        end (str): Ending city
        
    Returns:
        tuple: (path, distance)
    """
    path = nx.dijkstra_path(graph, start, end, weight='weight')
    distance = nx.dijkstra_path_length(graph, start, end, weight='weight')
    return path, distance

def solve_tsp_with_dijkstra(distances, cities):
    """
    Solve TSP by trying all possible permutations and using Dijkstra's algorithm
    for path finding between cities.
    
    Returns:
        tuple: (best_path, shortest_distance)
    """
    # Try all possible city orderings (except starting city which is fixed)
    start_city = cities[0]
    other_cities = cities[1:]
    
    best_path = None
    shortest_distance = float('inf')
    graph = create_graph(distances, cities)

    # Try each permutation of cities (excluding start city)
    for perm in permutations(other_cities):
        current_path = [start_city]
        current_distance = 0
        current_city = start_city
        
        # Go through each city in the permutation
        for next_city in perm:
            # Find shortest path to next city using Dijkstra's
            path_segment, segment_distance = dijkstra_shortest_path(graph,
                current_city, next_city)
            
            # Add path segment (excluding first city as it's already included)
            current_path.extend(path_segment[1:])
            current_distance += segment_distance
            current_city = next_city
        
        # Return to start city
        final_segment, final_distance = dijkstra_shortest_path(graph,
            current_city, start_city)
        current_path.extend(final_segment[1:])
        current_distance += final_distance
        
        # Update best path if current path is shorter
        if current_distance < shortest_distance:
            shortest_distance = current_distance
            best_path = current_path
    
    return best_path, shortest_distance

def main():
    """
    Main function to demonstrate TSP solution using Dijkstra's algorithm.
    """
    # Create sample problem
    distances, cities = create_tsp_graph(7)
    
    # Show initial graph
    print("Initial Graph:")
    visualize_graph(distances, cities)
    
    # Solve TSP
    print("\nSolving TSP using Dijkstra's algorithm...")
    best_path, shortest_distance = solve_tsp_with_dijkstra(distances, cities)
    
    # Print results
    print(f"\nResults:")
    print(f"Optimal path: {' -> '.join(best_path)}")
    print(f"Total distance: {shortest_distance}")
    
    # Visualize solution
    print("\nOptimal Path Visualization:")
    visualize_graph(distances, cities, best_path)

if __name__ == "__main__":
    main()