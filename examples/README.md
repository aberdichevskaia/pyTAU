# TAU Genetic Optimizer Example

This example demonstrates how to use the `GeneticOptimizer` class from the `tau` library to find the best partition of a graph using a genetic algorithm.

## Prerequisites

Before running this example, ensure that you have installed the necessary Python packages:

- `igraph`
- `networkx`
- `numpy`
- `tau` (the library containing the `GeneticOptimizer` class)

## Usage
The script can be run from the command line with the following options:

- `--graph`: Path to the graph file in adjacency list format (required).
- `--population_size`: Size of the population for the genetic algorithm (optional, default: 60).
- `--workers`: Number of worker threads to use (at least 2, default: number of available CPUs).
- `--max_generations`: Maximum number of generations to run the genetic algorithm (optional, default: 500).

## Example Command
```python
python example.py --graph example.graph --population_size 100 --workers 4 --max_generations 1000
```

This command will run the genetic optimization on the example graph located in this directory with a population size of 100, using 4 worker threads, and running for a maximum of 1000 generations.

## Output

The script will output a file named `TAU_partition_<graph_name>.npy`, which contains the best partition found by the genetic algorithm in the form of a NumPy array.