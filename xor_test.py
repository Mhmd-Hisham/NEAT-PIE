#!/usr/bin/env python3

from pprint import pprint

from NEATGenome import NEATGenome
from InnovationTracker import InnovationTracker
from NEATPopulation import NEATPopulation

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

innv_tracker = InnovationTracker()
initial_genome = NEATGenome.FullFeedForwardNetwork((2,1),
                                                    xor_fitness_function,
                                                    innv_tracker)

population = NEATPopulation(initial_genome)

accuracy = 0
max_fitness = 16.0

# Evolve while accuracy is less than 95 percent
while (accuracy < 95.0):
    population.apply_simulation(dataset)
    population.evolve()
    print(f"Generation: {population.current_generation}, " +
          f"Accuracy={accuracy}%",
          flush=True)

    accuracy = round(100*population.top_genome.fitness/max_fitness, 2)

pprint(population.top_genome.nodes)
pprint(population.top_genome.connections)

# uncomment if you have graphviz installed [pip install graphviz]
# you can see a visualization for the network

# from NNVisualizer import NNVisualizer
# NNVisualizer(population.top_genome, "Top Genome").view()

#
