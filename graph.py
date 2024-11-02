import networkx as nx
import matplotlib.pyplot as plt
import os

from datetime import datetime
from matplotlib.patches import FancyArrowPatch
from matplotlib.pyplot import gca

def visualize_graph(distances, cities, path=None, save=False):
    """
    Visualize TSP graph with path highlighting.
    """
    plt.figure(figsize=(10, 8))
    
    G = nx.Graph()
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            G.add_edge(cities[i], cities[j], weight=distances[i][j])
    
    pos = nx.circular_layout(G)
    
    # Draw all edges in light gray
    nx.draw_networkx_edges(G, pos, 
                           edge_color='lightgray',
                           width=1,
                           style='dashed')
    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, 
                                 edge_labels=edge_labels,
                                 font_size=10)
    
    if path is not None:
        # Draw solution path with arrows
        for i in range(len(path) - 1):
            start, end = path[i], path[i + 1]
            start_pos = pos[start]
            end_pos = pos[end]
            
            # Draw arrow manually
            arrow = FancyArrowPatch(start_pos, end_pos,
                                    arrowstyle='->',
                                    color='red',
                                    mutation_scale=15,
                                    linewidth=2)
            gca().add_patch(arrow)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                           node_color='lightblue',
                           node_size=1000,
                           edgecolors='black',
                           linewidths=2)
    
    nx.draw_networkx_labels(G, pos,
                            font_size=14,
                            font_weight='bold')
    
    if path is None:
        plt.title("Initial TSP Graph")
    else:
        total_distance = sum(distances[cities.index(path[i])][cities.index(path[i+1])] 
                             for i in range(len(path)-1))
        plt.title(f"TSP Solution Path\nPath: {' → '.join(path)}\nTotal Distance: {total_distance}")
    
    plt.axis('off')
    
    # Add legend
    if path is not None:
        legend_elements = [
            plt.Line2D([0], [0], color='lightgray', linestyle='--',
                       label='Available Paths'),
            plt.Line2D([0], [0], color='red', label='Solution Path')
        ]
        plt.legend(handles=legend_elements, loc='lower center',
                   bbox_to_anchor=(0.5, -0.15))
    
    plt.tight_layout()
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        savefig_name = f"qaoa_{len(cities)}_nodes_{timestamp}.png"
        savefig_name = os.path.join("graphs", savefig_name)
        plt.savefig(savefig_name, format="png")
    else:
        plt.show()
