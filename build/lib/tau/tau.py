from multiprocessing import Pool
from sklearn.metrics.cluster import pair_confusion_matrix
import itertools
import numpy as np
import os
import random
import time

from .partition import Partition

def _flip_coin():
    return random.uniform(0, 1) > .5


def _get_probabilities(values, selection_power):
    p = []
    values = np.max(values) + 1 - values
    denom = np.sum(values ** selection_power, dtype=np.int64)
    for value in values:
        p.append(value ** selection_power / denom)
    return p


def _compute_partition_similarity(partition_a, partition_b):
    conf = pair_confusion_matrix(partition_a, partition_b)
    b, d, c, a = conf.flatten()
    jac = a/(a+c+d)
    return jac


def _overlap(partition_memberships):
    consensus_partition = partition_memberships[0]
    n_nodes = len(consensus_partition)
    for i, partition in enumerate(partition_memberships[1:]):
        partition = partition
        cluster_id_mapping = {}
        c = 0
        for node_id in range(n_nodes):
            cur_pair = (consensus_partition[node_id], partition[node_id])
            if cur_pair not in cluster_id_mapping:
                cluster_id_mapping[cur_pair] = c
                c += 1
            consensus_partition[node_id] = cluster_id_mapping[cur_pair]
    return consensus_partition


def _create_partition(G_ig, sample_fraction):
        return Partition(G_ig, sample_fraction=sample_fraction)

def _single_crossover(G_ig, membership1, membership2):
        partitions_overlap = _overlap([membership1, membership2])
        single_offspring = Partition(G_ig, init_partition=partitions_overlap)
        return single_offspring

class GeneticOptimizer:
    def __init__(self, G_ig, population_size=60, n_workers=-1, 
                max_generations=500, selection_power=5, p_elite=0.1,
                p_immigrants=0.15, stopping_criterion_generations=10,
                stopping_criterion_jaccard=0.98, elite_similarity_threshold=0.9, 
                logging=False):
        self.G_ig = G_ig
        self.POPULATION_SIZE = max(10, population_size)
        cpus = os.cpu_count()
        self.N_WORKERS = min(cpus, self.POPULATION_SIZE) if n_workers == -1 else np.min(
                                                [cpus, self.POPULATION_SIZE, n_workers])
        self.N_ELITE = int(p_elite * self.POPULATION_SIZE)
        self.N_IMMIGRANTS = int(p_immigrants * self.POPULATION_SIZE)
        self.MAX_GENERATIONS = max_generations
        self.PROBS = _get_probabilities(np.arange(self.POPULATION_SIZE), selection_power)
        self.stopping_criterion_generations = stopping_criterion_generations
        self.stopping_criterion_jaccard = stopping_criterion_jaccard
        self.elite_similarity_threshold = elite_similarity_threshold 

        if logging == True:
            print(f'Main parameter values: pop_size={self.POPULATION_SIZE}, workers={self.N_WORKERS}, max_generations={self.MAX_GENERATIONS}')

    def __create_population(self, size_of_population):
        sample_fraction_per_indiv = np.random.uniform(.2, .9, size=size_of_population)
        params = [sample_fraction for sample_fraction in sample_fraction_per_indiv]
        pool = Pool(min(size_of_population, self.N_WORKERS))
        results = [pool.apply_async(_create_partition, (self.G_ig, sample_fraction,)) for sample_fraction in params]
        pool.close()
        pool.join()
        population = [x.get() for x in results]
        return population

    def __pop_crossover(self, n_offspring):
        idx_to_cross = []
        as_is_offspring = []
        for i in range(n_offspring):
            idx1, idx2 = np.random.choice(len(pop), size=2, replace=False, p=self.PROBS)
            if _flip_coin():
                idx_to_cross.append([idx1, idx2])
            else:
                as_is_offspring.append(pop[idx1])
        pool = Pool(self.N_WORKERS)
        results = [pool.apply_async(_single_crossover, (self.G_ig, pop[idx1].membership, pop[idx2].membership)) for idx1, idx2 in idx_to_cross]
        pool.close()
        pool.join()
        crossed_offspring = [x.get() for x in results]
        offspring = crossed_offspring + as_is_offspring
        return offspring

    def __selection_helper_compute_similarities(self, combinations):
        assert 0 < len(combinations) <= self.N_WORKERS
        pool = Pool(len(combinations))
        results = [pool.apply_async(_compute_partition_similarity, (pop[idx1].membership, pop[idx2].membership))
                for idx1, idx2 in combinations]
        pool.close()
        pool.join()
        similarities = [x.get() for x in results]
        similarities_dict = {tuple(sorted((idx1, idx2))): similarities[i] for i, (idx1, idx2) in enumerate(combinations)}
        return similarities_dict

    def __selection_helper_get_batch_of_pairs(self, elite_indices, candidate_idx, batch_size, computed=None):
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
        elite_indices, candidate_idx = [0], 1
        pairs_to_compute = self.__selection_helper_get_batch_of_pairs(elite_indices=elite_indices, candidate_idx=candidate_idx,
                                                            batch_size=self.N_WORKERS, computed=[])
        similarities_between_solutions = self.__selection_helper_compute_similarities(pairs_to_compute)
        computation_cycle_i, max_cycles = 1, 2
        computed_pairs = []
        while len(elite_indices) < self.N_ELITE and candidate_idx < len(pop):
            if computation_cycle_i == max_cycles:
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
                jac = similarities_between_solutions[elite_idx, candidate_idx]
                if jac > similarity_threshold:
                    elite_flag = False
                    break
            if elite_flag:
                elite_indices.append(candidate_idx)
            candidate_idx += 1
        return elite_indices

    def find_partition(self, logging=False):
        global pop
        last_best_memb = []
        best_modularity_per_generation = []
        cnt_convergence = 0

        # Population initialization
        pop = self.__create_population(size_of_population=self.POPULATION_SIZE)

        for generation_i in range(1, self.MAX_GENERATIONS+1):
            start_time = time.time()

            # Population optimization
            pool = Pool(self.N_WORKERS)
            results = [pool.apply_async(indiv.optimize, ()) for indiv in pop]
            pool.close()
            pool.join()
            pop = [x.get() for x in results]

            pop_fit = [indiv.fitness for indiv in pop]
            best_score = np.max(pop_fit)
            best_modularity_per_generation.append(best_score)
            best_indiv = pop[np.argmax(pop_fit)]

            # stopping criteria related
            if last_best_memb:
                sim_to_last_best = _compute_partition_similarity(best_indiv.membership, last_best_memb)
                if sim_to_last_best > self.stopping_criterion_jaccard:
                    cnt_convergence += 1
                else:
                    cnt_convergence = 0
            last_best_memb = best_indiv.membership
            if cnt_convergence == self.stopping_criterion_generations:
                break
            pop_rank_by_fitness = np.argsort(pop_fit)[::-1]
            pop = [pop[i] for i in pop_rank_by_fitness]
            if generation_i == self.MAX_GENERATIONS:
                break

            # elitist selection
            elite_idx = self.__elitist_selection(self.elite_similarity_threshold)
            elite = [pop[i] for i in elite_idx]

            # crossover
            n_offspring = self.POPULATION_SIZE-self.N_ELITE-self.N_IMMIGRANTS
            offspring = self.__pop_crossover(n_offspring)

            # immigration
            immigrants = self.__create_population(size_of_population=self.N_IMMIGRANTS)

            # mutation
            pool = Pool(min(len(offspring), self.N_WORKERS))
            results = [pool.apply_async(indiv.mutate, ()) for indiv in offspring]
            pool.close()
            pool.join()
            offspring = [x.get() for x in results]
            if logging == True:
                print(f'Generation {generation_i} Top fitness: {np.round(best_score, 6)}; Average fitness: '
                    f'{np.round(np.mean(pop_fit), 6)}; Time per generation: {np.round(time.time() - start_time, 2)}; '
                    f'convergence: {cnt_convergence}')
            pop = elite + offspring + immigrants

        # return best and modularity history
        return pop[0], best_modularity_per_generation
