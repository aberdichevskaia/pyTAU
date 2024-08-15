import igraph as ig
import numpy as np
import random

def _flip_coin():
    """ Simulate a coin flip, returning True for heads (50% probability). """
    return random.uniform(0, 1) > .5


class Partition:
    def __init__(self, G_ig, sample_fraction=.5, init_partition=None):
        """
        Initialize a Partition object.

        Parameters:
        G_ig (igraph.Graph): The input graph.
        sample_fraction (float): Fraction of the graph to sample (default is 0.5).
        init_partition (list): Initial partition membership (optional).
        """
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
        """ Initialize partition by sampling nodes or edges and applying Leiden algorithm."""
        if _flip_coin():
            # Sample nodes
            subsample = np.random.choice(self.n_nodes, size=self.sample_size_nodes, replace=False)
            subgraph = self.G_ig.subgraph(subsample)
        else:
            # Sample edges
            subsample = np.random.choice(self.n_edges, size=self.sample_size_edges, replace=False)
            subgraph = self.G_ig.subgraph_edges(subsample)

        # Initialize partition memberships
        subsample_partition_memb = np.zeros(self.n_nodes) - 1
        subsample_nodes = [v.index for v in subgraph.vs]

        # Apply Leiden algorithm on the subgraph
        subsample_subpartition = subgraph.community_leiden(objective_function='modularity')
        subsample_subpartition_memb = subsample_subpartition.membership

        # Assign membership for sampled nodes
        subsample_partition_memb[subsample_nodes] = subsample_subpartition_memb
        first_available_comm_id = np.max(subsample_subpartition_memb) + 1
        
        # Assign unassigned nodes to new communities
        arg_unassigned = subsample_partition_memb == -1
        subsample_partition_memb[arg_unassigned] = list(range(first_available_comm_id,
                                                              first_available_comm_id + sum(arg_unassigned)))
        self.membership = subsample_partition_memb.astype(int)
        self.n_comms = np.max(self.membership)+1

    def optimize(self):
        """
        Optimize the partition using the Leiden algorithm.
        
        Returns:
        Partition: The optimized partition object.
        """
        partition = self.G_ig.community_leiden(
            objective_function='modularity', 
            initial_membership=self.membership,
            n_iterations=3
        )
        self.membership = partition.membership
        self.n_comms = np.max(self.membership) + 1
        self.fitness = partition.modularity
        return self

    def __newman_split(self, indices, comm_id_to_split):
        """ Split a community using the Newman leading eigenvector method."""
        subgraph = self.G_ig.subgraph(indices)
        new_assignment = subgraph.community_leading_eigenvector(clusters=2).membership
        new_assignment[new_assignment == 0] = comm_id_to_split
        new_assignment[new_assignment == 1] = self.n_comms
        self.membership[self.membership == comm_id_to_split] = new_assignment

    def __random_split(self, indices):
        """Randomly split a community into two parts."""
        size_to_split = min(1, np.random.choice(len(indices)//2))
        idx_to_split = np.random.choice(indices, size=size_to_split, replace=False)
        self.membership[idx_to_split] = self.n_comms

    def __random_merge(self):
        """
        Randomly merge two connected communities.
        
        This method attempts to merge two different communities that are connected by an edge.
        It does so by selecting 10 random edges and checking if the communities at the 
        ends of those edges are different. If a pair of communities is found, they are merged, 
        and the process stops.
        """
        candidate_edges = random.choices(self.G_ig.es, k=10) # Select 10 random edges
        for edge in candidate_edges:
            v1, v2 = edge.tuple
            comm1, comm2 = self.membership[v1], self.membership[v2]
            if comm1 == comm2:
                continue # Skip if both vertices are in the same community
            # Merge community 1 into community 2
            self.membership[self.membership == comm1] = comm2
            # Reassign the highest community ID (self.n_comms - 1) to the old community ID
            self.membership[self.membership == self.n_comms-1] = comm1
            # Decrease the total number of communities by one
            self.n_comms -= 1
            break # Stop after merging one pair of communities

    def mutate(self):
        """
        Mutate the partition by either splitting or merging communities.

        Returns:
        Partition: The mutated partition object.
        """
        # Convert membership list to a numpy array for easier manipulation
        self.membership = np.array(self.membership)
        if _flip_coin():
            # Split a community
            comm_id_to_split = np.random.choice(self.n_comms)
            idx_to_split = np.where(self.membership == comm_id_to_split)[0]

            # Ensure the community is large enough to split
            if len(idx_to_split) > 2:
                min_comm_size_newman = 10

                # If the community is large enough, decide on the split method
                if len(idx_to_split) > min_comm_size_newman:
                    if _flip_coin():
                        # Use the Newman leading eigenvector method to split the community
                        self.__newman_split(idx_to_split, comm_id_to_split)
                    else:
                        # Perform a random split of the community
                        self.__random_split(idx_to_split)
                else:
                    # If the community is smaller, just do a random split
                    self.__random_split(idx_to_split)

                # Update the total number of communities
                self.n_comms += 1
        else:
            # Try to rndomly merge two connected communities
            # This reduces the total number of communities by one if successful 
            self.__random_merge()
        return self
