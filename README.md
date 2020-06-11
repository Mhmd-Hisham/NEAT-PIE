# NEAT-PIE

[Official Paper by Stanley & Miikkulainen](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)

A humple implementation of the NeuroEvolution of Augmenting Topologies[NEAT] algorithm written purely in Python3.


## Example Usage
### Evolving a Neural Network for XOR function:
```python3
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
```

A possible output:
```
Generation: 1, Accuracy=0%
Generation: 2, Accuracy=19.8%
Generation: 3, Accuracy=55.96%
Generation: 4, Accuracy=55.96%
Generation: 5, Accuracy=55.96%
Generation: 6, Accuracy=55.96%
Generation: 7, Accuracy=55.96%
Generation: 8, Accuracy=56.19%
Generation: 9, Accuracy=56.19%
Generation: 10, Accuracy=56.19%
Generation: 11, Accuracy=56.19%
Generation: 12, Accuracy=66.13%
Generation: 13, Accuracy=85.21%
Generation: 14, Accuracy=85.21%
Generation: 15, Accuracy=92.76%
Generation: 16, Accuracy=92.76%
Generation: 17, Accuracy=93.24%
Generation: 18, Accuracy=95.88%
{0: NodeGene(Node.Bias, 0, 0, '0'),
 1: NodeGene(Node.Sensor, 1, 0, '1'),
 2: NodeGene(Node.Sensor, 2, 0, '2'),
 3: NodeGene(Node.Output, 3, 2, '3'),
 5: NodeGene(Node.Hidden, 5, 1, '0000001:0000003')}
{'0000000:0000003': ConnectionGene(0, 3, -1.0, 2, False),
 '0000000:0000005': ConnectionGene(0, 5, -0.749092, 6, False),
 '0000001:0000003': ConnectionGene(1, 3, -2.205837, 0, False),
 '0000001:0000005': ConnectionGene(1, 5, 4, 7, False),
 '0000002:0000003': ConnectionGene(2, 3, -1.809044, 1, False),
 '0000002:0000005': ConnectionGene(2, 5, 4, 12, False),
 '0000005:0000003': ConnectionGene(5, 3, 4, 8, False)}
```

## Meta

Mohamed Hisham – [G-Mail](mailto:Mohamed00Hisham@Gmail.com) | [GitHub](https://github.com/Mhmd-Hisham) | [LinkedIn](https://www.linkedin.com/in/Mhmd-Hisham/)


This project is licensed under the GNU GPLv3 License - check [LICENSE](https://github.com/Mhmd-Hisham/NEAT-PIE/blob/master/LICENSE) for more details.

