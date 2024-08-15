from multiprocessing import Pool
from sklearn.metrics import adjusted_rand_score
import itertools
import numpy as np
import os
import random
import time

from .partition import Partition

def _flip_coin():
    """ Simulate a coin flip, returning True for heads (50% probability). """
    return random.uniform(0, 1) > .5


def _get_probabilities(values, selection_power):
    """
    Calculate selection probabilities based on fitness values.

    The higher the fitness value, the higher the selection probability.
    This method transforms fitness values to be proportional to their selection probabilities.
    
    Parameters:
    values (numpy.array): Array of fitness values.
    selection_power (float): Power to raise the fitness values to adjust selection pressure.
    
    Returns:
    list: A list of probabilities corresponding to each fitness value.
    """
    p = []
    values = np.max(values) + 1 - values
    denom = np.sum(values ** selection_power, dtype=np.int64)
    for value in values:
        # Calculate probability for each value based on selection power
        p.append(value ** selection_power / denom)
    return p


def _overlap(partition_memberships):
    """
    Generate a consensus partition by overlapping memberships (partitions).

    This function combines multiple partitions into a single consensus partition 
    by ensuring that nodes are consistently assigned to clusters across different partitions.

    Parameters:
    partition_memberships (list): A list of partitions (each being a list of cluster assignments).

    Returns:
    list: A consensus partition where each node is assigned a cluster based on overlapping.
    """
    consensus_partition = partition_memberships[0]
    n_nodes = len(consensus_partition)
    for partition in partition_memberships[1:]:
        # Map cluster IDs between partitions to ensure consistency
        partition = partition
        cluster_id_mapping = {}
        new_cluster_id = 0
        for node_id in range(n_nodes):
            cur_pair = (consensus_partition[node_id], partition[node_id])
            if cur_pair not in cluster_id_mapping:
                # Assign a new cluster ID if this pair of assignments hasn't been seen
                cluster_id_mapping[cur_pair] = new_cluster_id
                new_cluster_id += 1
            
            # Update the consensus partition with the mapped cluster ID
            consensus_partition[node_id] = cluster_id_mapping[cur_pair]
    return consensus_partition


def _create_partition(G_ig, sample_fraction):
    """
    Create a partition object with a sampled fraction of the graph.

    Parameters:
    G_ig (igraph.Graph): The input graph.
    sample_fraction (float): Fraction of the graph to sample for partitioning.

    Returns:
    Partition: A partition object based on the sampled graph.
    """
    return Partition(G_ig, sample_fraction=sample_fraction)

def _single_crossover(G_ig, membership1, membership2):
    """
    Perform a single-point crossover between two partitions.

    This function overlaps two partitions and generates a new partition 
    based on the overlap of the input partitions.

    Parameters:
    G_ig (igraph.Graph): The input graph.
    membership1 (list): Cluster membership of the first partition.
    membership2 (list): Cluster membership of the second partition.

    Returns:
    Partition: A new partition resulting from the crossover.
    """
    partitions_overlap = _overlap([membership1, membership2])
    single_offspring = Partition(G_ig, init_partition=partitions_overlap)
    return single_offspring

class GeneticOptimizer:
    def __init__(self, G_ig, population_size=60, n_workers=-1, 
                max_generations=500, selection_power=5, p_elite=0.1,
                p_immigrants=0.15, stopping_criterion_generations=10,
                stopping_criterion_similarity=0.98, elite_similarity_threshold=0.9, 
                logging=False):
        """
        Initialize the GeneticOptimizer with parameters for genetic optimization.

        Parameters:
        - G_ig (igraph.Graph): The input graph.
        - population_size (int): Number of individuals in the population.
        - n_workers (int): Number of workers for parallel processing (-1 to use all available CPUs).
        - max_generations (int): Maximum number of generations to run the optimization.
        - selection_power (float): Power used to adjust selection probabilities.
        - p_elite (float): Proportion of the population to be preserved as elites.
        - p_immigrants (float): Proportion of the population to be replaced by immigrants.
        - stopping_criterion_generations (int): Number of generations to check for convergence.
        - stopping_criterion_similarity (float): Similarity threshold for stopping criterion.
        - elite_similarity_threshold (float): Threshold for selecting diverse elite individuals.
        - logging (bool): Flag to enable or disable logging of generation statistics.
        """
        self.G_ig = G_ig
        self.POPULATION_SIZE = max(10, population_size)

        assert n_workers == -1 or n_workers >= 2, (
            "Number of used workers must be at least 2. "
            "Pass -1 to use all available workers."
        )
        cpus = os.cpu_count()
        self.N_WORKERS = min(cpus, self.POPULATION_SIZE) if n_workers == -1 else np.min(
                                                [cpus, self.POPULATION_SIZE, n_workers])
        self.N_ELITE = int(p_elite * self.POPULATION_SIZE)
        self.N_IMMIGRANTS = int(p_immigrants * self.POPULATION_SIZE)
        self.MAX_GENERATIONS = max_generations
        self.PROBS = _get_probabilities(np.arange(self.POPULATION_SIZE), selection_power)
        self.stopping_criterion_generations = stopping_criterion_generations
        self.stopping_criterion_similarity = stopping_criterion_similarity
        self.elite_similarity_threshold = elite_similarity_threshold 

        if logging == True:
            print(f'Main parameter values: pop_size={self.POPULATION_SIZE}, workers={self.N_WORKERS}, max_generations={self.MAX_GENERATIONS}')

    def __create_population(self, size_of_population):
        """
        Create a population of partitions with random sample fractions.

        Parameters:
        size_of_population (int): Size of the population to create.

        Returns:
        list: A list of Partition objects representing the population.
        """
        sample_fraction_per_indiv = np.random.uniform(.2, .9, size=size_of_population)
        params = [sample_fraction for sample_fraction in sample_fraction_per_indiv]
        with Pool(min(size_of_population, self.N_WORKERS)) as pool:
            # Use multiprocessing to create partitions in parallel
            results = [pool.apply_async(_create_partition, (self.G_ig, sample_fraction,)) for sample_fraction in params]
        population = [x.get() for x in results]
        return population

    def __pop_crossover(self, n_offspring):
        """
        Perform crossover on selected individuals to generate offspring.

        Parameters:
        n_offspring (int): Number of offspring to generate.

        Returns:
        list: A list of new Partition objects (offspring) generated by crossover.
        """
        idx_to_cross = []
        as_is_offspring = []
        for i in range(n_offspring):
            idx1, idx2 = np.random.choice(len(pop), size=2, replace=False, p=self.PROBS)
            if _flip_coin():
                # If the coin flip is True, add the pair to the crossover list
                idx_to_cross.append([idx1, idx2])
            else:
                # Otherwise, keep one parent unchanged (offspring is a `clone` of a parent)
                as_is_offspring.append(pop[idx1])
        with Pool(self.N_WORKERS) as pool:
            # Perform crossover in parallel using multiprocessing
            results = [pool.apply_async(_single_crossover, (self.G_ig, pop[idx1].membership, pop[idx2].membership)) for idx1, idx2 in idx_to_cross]
        crossed_offspring = [x.get() for x in results]
        offspring = crossed_offspring + as_is_offspring
        return offspring

    def __selection_helper_compute_similarities(self, combinations):
        """
        Compute the similarities between pairs of individuals.

        Parameters:
        combinations (list): List of tuples representing pairs of indices to compare.

        Returns:
        dict: A dictionary with tuples of indices as keys and similarity scores as values.
        """
        assert 0 < len(combinations) <= self.N_WORKERS
        with Pool(len(combinations)) as pool:
            results = [pool.apply_async(adjusted_rand_score, (pop[idx1].membership, pop[idx2].membership))
                for idx1, idx2 in combinations]
        similarities = [x.get() for x in results]
        similarities_dict = {tuple(sorted((idx1, idx2))): similarities[i] for i, (idx1, idx2) in enumerate(combinations)}
        return similarities_dict

    def __selection_helper_get_batch_of_pairs(self, elite_indices, candidate_idx, batch_size, computed=None):
        """
        Generate a batch of pairs of individuals for similarity computation.

        Parameters:
        elite_indices (list): List of indices of elite individuals.
        candidate_idx (int): Index of the candidate individual.
        batch_size (int): Maximum size of the batch.
        computed (list): List of already computed pairs (optional).

        Returns:
        list: A list of pairs of indices to compute similarities for.
        """
        pairs = list(itertools.product(elite_indices, [candidate_idx]))
        pairs = [pair for pair in pairs if pair not in computed] if computed is not None else pairs
        batch_overflow = False if len(pairs) < batch_size else True
        i = 1
        while not batch_overflow:
            pairs += list(itertools.product(elite_indices, [candidate_idx+i]))
            for j in range(i):
                pairs.append((candidate_idx+j, candidate_idx+i))
            batch_overflow = False if len(pairs) < batch_size else True
            i += 1
        return pairs[:batch_size]

    def __elitist_selection(self, similarity_threshold):
        """
        Perform elitist selection by selecting the most diverse individuals
        from the population. The goal is to select a set of elite individuals
        that are not too similar to each other based on the provided similarity threshold.
         
        The method works as follows:
        1. Start with the best individual as the first elite member.
        2. Iteratively compare each subsequent individual to the already selected elites.
        3. If an individual is sufficiently different from all current elites (based on the similarity threshold),
           it is added to the elite set.
        4. If the number of elite individuals is below the required amount and no more diverse individuals can be found,
           randomly select the remaining individuals to complete the elite set.
        5. Return the indices of the selected elite individuals.
        """
        elite_indices, candidate_idx = [0], 1
        pairs_to_compute = self.__selection_helper_get_batch_of_pairs(elite_indices=elite_indices, candidate_idx=candidate_idx,
                                                            batch_size=self.N_WORKERS, computed=[])
        # Compute similarities between candidate and elite partitions
        similarities_between_solutions = self.__selection_helper_compute_similarities(pairs_to_compute)
        computation_cycle_i, max_cycles = 1, 2
        computed_pairs = []
        while len(elite_indices) < self.N_ELITE and candidate_idx < len(pop):
            if computation_cycle_i == max_cycles:
                # If maximum iterations of selection reached, randomly select remaining elites
                n_remaining = self.N_ELITE - len(elite_indices)
                elite_indices += list(np.random.choice(np.arange(candidate_idx, len(pop)), size=n_remaining, replace=False))
                break
            elite_flag = True
            for elite_idx in elite_indices:
                if (elite_idx, candidate_idx) not in similarities_between_solutions and computation_cycle_i < max_cycles:
                    computation_cycle_i += 1
                    computed_for_candidate = [(i, j) for (i, j) in computed_pairs
                                            if i >= candidate_idx or j >= candidate_idx]
                    new_pairs = self.__selection_helper_get_batch_of_pairs(elite_indices=elite_indices, candidate_idx=candidate_idx,
                                                                    batch_size=self.N_WORKERS, computed=computed_for_candidate)
                    similarities_between_solutions.update(self.__selection_helper_compute_similarities(new_pairs))
                    computed_pairs += new_pairs
                similarity_score = similarities_between_solutions[elite_idx, candidate_idx]
                if similarity_score > similarity_threshold:
                    elite_flag = False
                    break
            if elite_flag:
                elite_indices.append(candidate_idx)
            candidate_idx += 1
        return elite_indices

    def __log_generation(self, generation_i, best_score, pop_fit, start_time, cnt_convergence):
        """
        Log the details of the current generation during the optimization process.

        Parameters:
        generation_i (int): The index of the current generation.
        best_score (float): The fitness score of the best individual in the generation.
        pop_fit (list): A list of fitness scores for the population.
        start_time (float): The start time of the generation (for timing purposes).
        cnt_convergence (int): The current count of convergence generations.
        """
        print(f'Generation {generation_i} Top fitness: {np.round(best_score, 6)}; Average fitness: '
                f'{np.round(np.mean(pop_fit), 6)}; Time per generation: {np.round(time.time() - start_time, 2)}; '
                f'convergence: {cnt_convergence}')

    def find_partition(self, logging=False):
        """
        Run the genetic optimization process to find the best partition.

        This method initializes the population, then iteratively optimizes it through selection,
        crossover, mutation, and immigration over multiple generations. The process stops when
        the maximum number of generations is reached or when convergence criteria are met.

        Parameters:
        logging (bool): Whether to log the details of each generation.

        Returns:
        tuple: The best partition found and the history of modularity scores per generation.
        """
        global pop
        last_best_memb = []
        best_modularity_per_generation = []
        cnt_convergence = 0

        # Step 1: Initialize the population
        pop = self.__create_population(size_of_population=self.POPULATION_SIZE)

        for generation_i in range(1, self.MAX_GENERATIONS+1):
            start_time = time.time()

            # Step 2: Optimize the population
            with Pool(self.N_WORKERS) as pool:
                results = [pool.apply_async(indiv.optimize, ()) for indiv in pop]
            pop = [x.get() for x in results]

            pop_fit = [indiv.fitness for indiv in pop]
            best_score = np.max(pop_fit)
            best_modularity_per_generation.append(best_score)
            best_indiv = pop[np.argmax(pop_fit)]

            # Step 3: Check stopping criteria based on convergence
            # The convergence criterion is used to determine whether the optimization
            # process should be stopped early. It checks if the best individual in the 
            # current generation is too similar to the best individual from the previous 
            # generation(s). If this similarity exceeds a specified threshold, it indicates 
            # that the population may have converged to a stable solution.

            if last_best_memb:
                sim_to_last_best = _compute_partition_similarity(best_indiv.membership, last_best_memb)
                if sim_to_last_best > self.stopping_criterion_similarity:
                    cnt_convergence += 1
                else:
                    cnt_convergence = 0
            last_best_memb = best_indiv.membership
            if cnt_convergence == self.stopping_criterion_generations:
                break
            pop_rank_by_fitness = np.argsort(pop_fit)[::-1]
            pop = [pop[i] for i in pop_rank_by_fitness]
            if generation_i == self.MAX_GENERATIONS:
                if logging == True:
                    # Log the generation details if logging is enabled
                    self.__log_generation(generation_i, best_score, pop_fit, 
                    start_time, cnt_convergence)
                break

            # Step 4: Perform elitist selection, crossover, mutation, and immigration

            # 4.1 Elitist Selection:
            # Select the top individuals from the current population based on fitness.
            # These individuals are preserved without modification and carried over to 
            # the next generation. The goal is to maintain the best solutions found so far.
            elite_idx = self.__elitist_selection(self.elite_similarity_threshold)
            elite = [pop[i] for i in elite_idx]

            # 4.2 Crossover:
            # Generate new offspring by crossing over pairs of individuals selected from
            # the current population. The crossover process combines parts of two parents 
            # to create a child, promoting genetic diversity in the population.
            n_offspring = self.POPULATION_SIZE-self.N_ELITE-self.N_IMMIGRANTS
            offspring = self.__pop_crossover(n_offspring)

            # 4.3 Mutation:
            # Apply random mutations to the offspring to further enhance genetic diversity.
            # Mutation introduces small random changes to an individual's structure, which
            # can help escape local optima and explore new areas of the solution space.
            with Pool(min(len(offspring), self.N_WORKERS)) as pool:
                results = [pool.apply_async(indiv.mutate, ()) for indiv in offspring]
            offspring = [x.get() for x in results]

            # 4.3 Immigration:
            # Introduce new individuals into the population by creating a small number of
            # "immigrants." These are new partitions generated from scratch, which helps
            # in introducing fresh genetic material and preventing premature convergence.
            immigrants = self.__create_population(size_of_population=self.N_IMMIGRANTS)

            # Step 5: Combine the elite, mutated offspring, and immigrants to form the 
            # new population.
            pop = elite + offspring + immigrants

            if logging == True:
                # Log the generation details if logging is enabled
                self.__log_generation(generation_i, best_score, pop_fit, start_time, cnt_convergence)

        # Return the best individual and modularity history
        return pop[0], best_modularity_per_generation
