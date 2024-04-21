import igraph as ig
import numpy as np
import random

def _flip_coin():
    return random.uniform(0, 1) > .5


class Partition:
    def __init__(self, G_ig, sample_fraction=.5, init_partition=None):
        np.random.seed()
        self.G_ig = G_ig
        self.n_nodes = self.G_ig.vcount()
        self.n_edges = self.G_ig.ecount()
        self.sample_size_nodes = int(self.n_nodes * sample_fraction)
        self.sample_size_edges = int(self.n_edges * sample_fraction)
        self.membership = []
        self.n_comms = 0
        self.fitness = None
        if init_partition is None:
            self.__initialize_partition()
        else:
            self.membership = init_partition
            self.n_comms = len(np.unique(self.membership))

    def __initialize_partition(self):
        if _flip_coin():
            # sample nodes
            subsample = np.random.choice(self.n_nodes, size=self.sample_size_nodes, replace=False)
            subgraph = self.G_ig.subgraph(subsample)
        else:
            # sample edges
            subsample = np.random.choice(self.n_edges, size=self.sample_size_edges, replace=False)
            subgraph = self.G_ig.subgraph_edges(subsample)
        subsample_partition_memb = np.zeros(self.n_nodes) - 1
        subsample_nodes = [v.index for v in subgraph.vs]
        # leiden on subgraph
        subsample_subpartition = subgraph.community_leiden(objective_function='modularity')
        subsample_subpartition_memb = subsample_subpartition.membership
        subsample_partition_memb[subsample_nodes] = subsample_subpartition_memb
        first_available_comm_id = np.max(subsample_subpartition_memb) + 1
        arg_unassigned = subsample_partition_memb == -1
        subsample_partition_memb[arg_unassigned] = list(range(first_available_comm_id,
                                                              first_available_comm_id + sum(arg_unassigned)))
        self.membership = subsample_partition_memb.astype(int)
        self.n_comms = np.max(self.membership)+1

    def optimize(self):
        # leiden
        partition = self.G_ig.community_leiden(objective_function='modularity', initial_membership=self.membership,
                                          n_iterations=3)
        self.membership = partition.membership
        self.n_comms = np.max(self.membership) + 1
        self.fitness = partition.modularity
        return self

    def __newman_split(self, indices, comm_id_to_split):
        # newman
        subgraph = self.G_ig.subgraph(indices)
        new_assignment = subgraph.community_leading_eigenvector(clusters=2).membership
        new_assignment[new_assignment == 0] = comm_id_to_split
        new_assignment[new_assignment == 1] = self.n_comms
        self.membership[self.membership == comm_id_to_split] = new_assignment

    def __random_split(self, indices):
        size_to_split = min(1, np.random.choice(len(indices)//2))
        idx_to_split = np.random.choice(indices, size=size_to_split, replace=False)
        self.membership[idx_to_split] = self.n_comms

    def __random_merge(self):
        # randomly merge two connected communities
        candidate_edges = random.choices(self.G_ig.es, k=10)
        for i, e in enumerate(candidate_edges):
            v1, v2 = e.tuple
            comm1, comm2 = self.membership[v1], self.membership[v2]
            if comm1 == comm2:
                continue
            self.membership[self.membership == comm1] = comm2
            self.membership[self.membership == self.n_comms-1] = comm1
            self.n_comms -= 1
            break

    def mutate(self):
        self.membership = np.array(self.membership)
        if _flip_coin():
            # split a community
            comm_id_to_split = np.random.choice(self.n_comms)
            idx_to_split = np.where(self.membership == comm_id_to_split)[0]
            if len(idx_to_split) > 2:
                min_comm_size_newman = 10
                if len(idx_to_split) > min_comm_size_newman:
                    if _flip_coin():
                        self.__newman_split(idx_to_split, comm_id_to_split)
                    else:
                        self.__random_split(idx_to_split)
                else:
                    self.__random_split(idx_to_split)
                self.n_comms += 1
        else:
            # randomly merge two connected communities
            self.__random_merge()
        return self
