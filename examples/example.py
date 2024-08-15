import argparse
import igraph as ig
import itertools
import networkx as nx
import numpy as np
import os
import sys

from tau import GeneticOptimizer

def load_graph(path):
    """
    Load a graph from an adjacency list file using NetworkX and convert it to an igraph object.
    
    Parameters:
    path (str): The path to the file containing the adjacency list of the graph.
    
    Returns:
    ig.Graph: The loaded graph as an igraph object.
    """

     # Load the graph using NetworkX
    nx_graph = nx.read_adjlist(path)

    # Relabel nodes to have integer IDs
    mapping = {node: idx for idx, node in enumerate(nx_graph.nodes())}
    nx_graph = nx.relabel_nodes(nx_graph, mapping)

    # Convert the NetworkX graph to an igraph graph
    ig_graph = ig.Graph(len(nx_graph), list(zip(*list(zip(*nx.to_edgelist(nx_graph)))[:2])))
    
    return ig_graph


if __name__ == "__main__":
    # Parse script parameters
    parser = argparse.ArgumentParser(description='TAU')

    # General parameters
    parser.add_argument(
        '--graph', 
        type=str, 
        required=True, 
        help='Path to the graph file; supports adjacency list format'
    )

    parser.add_argument(
        '--population_size', 
        type=int, 
        default=60, 
        help='Size of the population; default is 60'
    )

    parser.add_argument(
        '--workers', 
        type=int, 
        default=-1, 
        help='Number of workers; should be at least 2; default is the number of available CPUs'
    )

    parser.add_argument(
        '--max_generations', 
        type=int, 
        default=500, 
        help='Maximum number of generations to run; default is 500'
    )                                                          
    
    # Parse arguments
    args = parser.parse_args()

    # Load the graph from the specified file
    graph = load_graph(args.graph)

    # Define parameters for the genetic algorithm
    elite_fraction = 0.1
    immigrants_fraction = 0.15
    stopping_criterion_generations = 10
    stopping_criterion_similarity = 0.98
    elite_similarity_threshold = 0.9
    selection_power = 5

    # Initialize the GeneticOptimizer with the specified parameters
    optimizer = GeneticOptimizer(
        G_ig=graph,
        population_size=args.population_size,
        n_workers=args.workers,
        max_generations=args.max_generations,
        selection_power=selection_power,
        p_elite=elite_fraction,
        p_immigrants=immigrants_fraction,
        stopping_criterion_generations=stopping_criterion_generations,
        stopping_criterion_similarity=stopping_criterion_similarity,
        elite_similarity_threshold=elite_similarity_threshold,
        logging=True
    )

    # Run the genetic optimization process to find the best partition
    best_partition, modularity_history = optimizer.find_partition(logging=True)
    
    # Save the best partition's membership to a file
    output_filename = f'TAU_partition_{os.path.basename(args.graph)}.npy'
    np.save(output_filename, best_partition.membership)
    