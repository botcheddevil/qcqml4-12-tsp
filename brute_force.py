from itertools import permutations

# Function to calculate the total distance of a given path
def calculate_distance(graph, path):
    distance = 0
    for i in range(len(path) - 1):
        distance += graph[path[i]][path[i + 1]]
    # Adding distance from last to first to complete the cycle
    distance += graph[path[-1]][path[0]]
    return distance

# Function to find the shortest path using brute-force
def tsp_brute_force(graph):
    n = len(graph)
    cities = range(n)
    
    # Initialize minimum distance as infinity and an empty path
    min_distance = float('inf')
    min_path = []
    
    # Check all permutations of cities starting from the first city (fixed as start)
    for perm in permutations(cities[1:]):  # Fix the first city as starting point
        path = [0] + list(perm)
        current_distance = calculate_distance(graph, path)
        
        if current_distance < min_distance:
            min_distance = current_distance
            min_path = path
    
    return min_path, min_distance

# Example graph represented as an adjacency matrix
graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

# Solve the TSP using brute-force
path, distance = tsp_brute_force(graph)
print("Shortest path:", path)
print("Minimum distance:", distance)