from itertools import permutations


def solve_tsp_with_bruteforce(distances, cities):
    """
    Solve TSP by trying all possible permutations and using direct distances between cities.
    
    Returns:
        tuple: (best_path, shortest_distance)
    """
    # Try all possible city orderings (except starting city which is fixed)
    start_city = cities[0]
    other_cities = cities[1:]
    
    best_path = None
    shortest_distance = float('inf')
    
    # Map city names to indices
    city_indices = {city: idx for idx, city in enumerate(cities)}
    
    # Try each permutation of cities (excluding start city)
    for perm in permutations(other_cities):
        current_path = [start_city] + list(perm) + [start_city]
        current_distance = 0
        # Sum up distances between consecutive cities
        for i in range(len(current_path)-1):
            city1 = current_path[i]
            city2 = current_path[i+1]
            idx1 = city_indices[city1]
            idx2 = city_indices[city2]
            current_distance += distances[idx1][idx2]
        
        # Update best path if current path is shorter
        if current_distance < shortest_distance:
            shortest_distance = current_distance
            best_path = current_path
    
    return best_path, shortest_distance