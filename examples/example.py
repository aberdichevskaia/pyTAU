import argparse
import igraph as ig
import itertools
import networkx as nx
import numpy as np
import os
import sys

from tau import GeneticOptimizer

def load_graph(path):
    nx_graph = nx.read_adjlist(path)
    mapping = dict(zip(nx_graph.nodes(), range(nx_graph.number_of_nodes())))
    nx_graph = nx.relabel_nodes(nx_graph, mapping)
    ig_graph = ig.Graph(len(nx_graph), list(zip(*list(zip(*nx.to_edgelist(nx_graph)))[:2])))
    return ig_graph


if __name__ == "__main__":
    # parse script parameters
    parser = argparse.ArgumentParser(description='TAU')
    # general parameters
    parser.add_argument('--graph', type=str, help='path to graph file; supports adjacency list format')
    parser.add_argument('--size', type=int, default=60, help='size of population; default is 60')
    parser.add_argument('--workers', type=int, default=-1, help='number of workers; '
                                                                'default is number of available CPUs')
    parser.add_argument('--max_generations', type=int, default=500, help='maximum number of generations to run;'
                                                                         ' default is 500')
    args = parser.parse_args()
    G_ig = load_graph(args.graph)
    p_elite = .1
    p_immigrants = .15
    stopping_criterion_generations = 10
    stopping_criterion_jaccard = .98
    elite_similarity_threshold = .9
    SELECTION_POWER = 5

    optimizer = GeneticOptimizer(G_ig, population_size=args.size, 
                        n_workers=args.workers, max_generations=args.max_generations,
                        logging=True)
    best_partition, mod_history = optimizer.find_partition(logging=True)
    np.save(f'TAU_partition_{args.graph}.npy', best_partition.membership)
    