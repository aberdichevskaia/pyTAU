[![PyPI version](https://img.shields.io/pypi/v/tau-graph-clustering.svg?logo=pypi)](https://pypi.org/project/tau-graph-clustering/)

# Tel Aviv University (TAU)

TAU ia a Python package that provides advanced graph clustering using a combination of the `igraph` library, the Leiden algorithm, and a genetic algorithm. It is designed to achieve superior modularity compared to the standard Leiden algorithm.

## Features

- **High Modularity:** Achieve better modularity scores than the Leiden algorithm alone.
- **Genetic Algorithm Optimization:** Utilizes a genetic algorithm to refine clustering results.
- **Parallel Processing:** Supports multi-threaded execution to speed up computations.

## Requirements

- Python version 3.8 or higher is required.
- For optimal performance, it is recommended to run on Linux or macOS. On Windows, performance may be significantly slower due to architectural differences.
- The tool requires a minimum of 2 threads to run correctly.
  

## Installation

The tool is available on PyPI and can be installed using pip:

```bash
pip install tau-graph-clustering
```

## Usage

### Importing the Tool

To use the tool, import the necessary classes and functions:

```python
from tau import GeneticOptimizer
```

### Creating a Clustering Instance

Create an instance of the `GeneticOptimizer` class with the following parameters:

- **\`G_ig\`**: The `igraph` graph object.
- **\`population_size\`** (default=60): The number of partitions in the genetic algorithm's population.
- **\`n_workers\`** (default=-1): The number of worker threads. Use -1 to utilize all available cores. The minimal requered number of workers is 2. 
- **\`max_generations\`** (default=500): The maximum number of generations for the genetic algorithm.
- **\`selection_power\`** (default=5): The power of selection pressure in the genetic algorithm.
- **\`p_elite\`** (default=0.1): The proportion of elite partitions to carry over to the next generation.
- **\`p_immigrants\`** (default=0.15): The proportion of new random partitions introduced each generation.
- **\`stopping_criterion_generations\`** (default=10): The number of generations without a significant improvement to stop the algorithm.
- **\`stopping_criterion_similarity\`** (default=0.98): The threshold for similarity considered insignificant, which increments the counter from the previous point.
- **\`elite_similarity_threshold\`** (default=0.9): The threshold for similarity among elite solutions; we select partitions as elite that are not more similar to each other than this value.
- **\`logging\`** (default=False): Enable logging for detailed process information.

Example:

```python
clustering = GeneticOptimizer(G_ig=my_graph, population_size=100, max_generations=200)
```

### Finding the Optimal Partition

Use the `find_partition` method to perform the clustering:

```python
partition = clustering.find_partition()
```

## Examples

For more examples and use cases, refer to the [examples](examples) directory.

## Citation

If you use TAU in your research, please cite:

> Gal Gilad, Roded Sharan, From Leiden to Tel-Aviv University (TAU): exploring clustering solutions via a genetic algorithm, PNAS Nexus, Volume 2, Issue 6, June 2023, pgad180, https://doi.org/10.1093/pnasnexus/pgad180. 

## License

This project is licensed under the terms of the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Contributions

Contributions are welcome! Please submit pull requests or open issues for any bugs or feature requests.

## Contact

For questions or support, please contact <berdichevskaya.a.g@gmail.com>.
