import numpy as np
import networkx as nx
import igraph as ig
import itertools
import argparse


# globals and hyper-parameters
G_ig = None
POPULATION_SIZE = 60
N_WORKERS = 60
MAX_GENERATIONS = 500
N_IMMIGRANTS = -1
N_ELITE = -1
SELECTION_POWER = 5
PROBS = []
pop = []
p_elite = .1
p_immigrants = .15
stopping_criterion_generations = 10
stopping_criterion_jaccard = .98
elite_similarity_threshold = .9


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

    # set global variable values
    g = load_graph(args.graph)
    population_size = max(10, args.size)
    cpus = os.cpu_count()
    N_WORKERS = min(cpus, population_size) if args.workers == -1 else np.min([cpus, population_size, args.workers])
    PROBS = get_probabilities(np.arange(population_size))
    N_ELITE, N_IMMIGRANTS = int(p_elite * population_size), int(p_immigrants * population_size)
    G_ig = g
    POPULATION_SIZE = population_size
    MAX_GENERATIONS = args.max_generations

    print(f'Main parameter values: pop_size={POPULATION_SIZE}, workers={N_WORKERS}, max_generations={MAX_GENERATIONS}')

    best_partition, mod_history = find_partition()
    np.save(f'TAU_partition_{args.graph}.npy', best_partition.membership)
    