#!/usr/bin/env python3

import random
import statistics
from pprint import pprint

from NEATGenome import NEATGenome
from InnovationTracker import InnovationTracker
from NEATPopulation import NEATPopulation
from NodeGene import Node

# seed the random number generator
random.seed(12345)

dataset = [({1:0, 2:0}, {3:0}),
           ({1:1, 2:0}, {3:1}),
           ({1:0, 2:1}, {3:1}),
           ({1:1, 2:1}, {3:0}),]

def xor_fitness_function(genome, dataset: list) -> float:
    """ Calculates and updates the fitness property of the genome..
        dataset is a list of tuples where the first item is the input dictionary,
        and the second is the expected output dictionary.
        returns the fitness value too..
    """
    # example fitness function for XOR
    # use the fitness_updated property if you don't need to calculate
    # the fitness more than once for a genome
    if (genome.fitness_updated):
        return genome.fitness

    total_diff = 0
    for nn_inputs, nn_outputs in dataset:
        prediction = genome.predict(nn_inputs)
        total_diff += sum(abs(nn_outputs[node]-prediction[node])
                                     for node in nn_outputs)

    return round((4-total_diff)**2, genome.precision)

def xor_test(max_generations: int=100, verbose: bool=True):
    innv_tracker = InnovationTracker()
    initial_genome = NEATGenome.FullFeedForwardNetwork((2,1),
                                                        xor_fitness_function,
                                                        innv_tracker)

    population = NEATPopulation(initial_genome)

    accuracy = 0
    max_fitness = 16.0

    # Evolve while accuracy is less than 95 percent or until max_generations
    while (accuracy < 95.0 and population.current_generation < max_generations):
        population.apply_simulation(dataset)
        population.evolve()
        if (verbose):
            print(f"Generation: {population.current_generation}, Accuracy={accuracy}%", flush=True)

        accuracy = round(100*population.top_genome.fitness/max_fitness, 2)

    return population.top_genome, population.current_generation

def xor_benchmark(n_runs=100):
    # run the test 100 times to benchmark the algorithm
    # From the official NEAT paper:
    # """ 
    #    On 100 runs, the first experiment shows that the NEAT system finds a structure for
    #    XOR in an average of 32 generations (4,755 networks evaluated, std 2,553). On average
    #    a solution network had 2.35 hidden nodes and 7.48 nondisabled connection genes. The
    #    number of nodes and connections is close to optimal considering that the smallest pos-
    #    sible network has a single hidden unit (Figure 5(b)). NEAT is very consistent in finding
    #    a solution. It did not fail once in 100 simulations. The worst performance took 13,459
    #    evaluations, or about 90 generations (compared to 32 generations on average). The
    #    standard deviation for number of nodes used in a solution was 1.11, meaning NEAT
    #    very consistently used 1 or 2 hidden nodes to build an XOR network. In conclusion,
    #    NEAT solves the XOR problem without trouble and in doing so keeps the topology
    #    small.
    # """
    n_hidden_nodes = []
    n_generations = []
    n_connections = []

    for _ in range(n_runs):
        genome, generation = xor_test(max_generations=100, verbose=False)
        hidden_nodes = [node for node in genome.nodes.values() if node.type == Node.Hidden]
        nondisabled_connections = [conn for conn in genome.connections.values() if not conn.disabled]
        n_hidden_nodes.append(len(hidden_nodes))
        n_generations.append(generation)
        n_connections.append(len(nondisabled_connections))

    print(f"Average number of hidden nodes: {sum(n_hidden_nodes)/n_runs}")
    print(f"Standard deviation: {statistics.stdev(n_hidden_nodes)}")
    print()
    print(f"Average number of generations: {sum(n_generations)/n_runs}")
    print(f"Standard deviation: {statistics.stdev(n_generations)}")
    print()
    print(f"Average number of connections: {sum(n_connections)/n_runs}")
    print(f"Standard deviation: {statistics.stdev(n_connections)}")
    print()
    print(f"Optimal solution found in {n_hidden_nodes.count(1)} of {n_runs} runs")

    return genome

if __name__ == "__main__":
    top_genome = xor_benchmark()
    
    # uncomment if you have graphviz installed [pip install graphviz]
    # you can see a visualization for the network

    # from NNVisualizer import NNVisualizer
    # NNVisualizer(top_genome, "Top Genome").view()
    