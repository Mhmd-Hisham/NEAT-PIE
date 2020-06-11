#!/usr/bin/env python3

import random

from InnovationTracker import InnovationTracker
from NEATConfiguration import NEATConfiguration as config
from NEATGenome import NEATGenome
from NEATSpecies import NEATSpecies

class NEATPopulation:
    """
        NEATPopulation class that stores the species in each generation.
    """
    fitenss_epsilon            = config.fitenss_epsilon
    population_size            = config.population_size
    mutate_only_chance         = config.mutate_only_chance
    interspecies_mating_chance = config.interspecies_mating_chance

    def __init__(self,
                 initial_genome: NEATGenome):
        """ Population Constructor
            initial_genome -> The initial genome to initialize the population with.
        """
        self.initial_genome     = NEATGenome.Copy(initial_genome)
        self.population_tracker = self.initial_genome.innv_tracker

        self.species = []

        for _ in range(self.population_size):
            self.add_genome(NEATGenome.Copy(self.initial_genome))

        # number of calls to the evolve function
        self.current_generation = 0

        # restores the top genome across all generations
        self.top_genome = NEATGenome.Copy(initial_genome)
        self.top_genome_generation = 0

    def get_random_genome(self, copy: bool=False) -> NEATGenome:
        """
            Returns a random gneome from a random species.
            if copy is set to true, returns a copy of the genome.
        """
        return random.choice(self.species).get_random_member(copy=copy)

    def get_fittest_genome(self):
        """
            Returns the fittest genome in all species in the population.
        """
        # get the top in each species
        tops = [species.get_fittest_member() for species in self.species]

        # return the top genome amongst all species
        return max(tops, key=lambda gnome: gnome.fitness)

    def apply_simulation(self, *args, **kwargs):
        """ Calculates the fitness and adjusted fitness for each
            genome in each species.
        """

        for species in self.species:
            species.update_members_adjusted_fitness(*args, **kwargs)

    def add_genome(self, genome: NEATGenome, copy: bool=False) -> None:
        """
            Adds the given genome to a proper species in the population,
            or creates a new species if it's not compatibile with all
            the current species.
        """
        if copy: genome = NEATGenome.Copy(genome)

        # search for a compatibile species for the genome
        predicate = lambda species: species.is_compatible(genome)
        species = next(filter(predicate, self.species), None)

        # if found, then add the genome to it
        if (species!=None):
            species.add_member(genome)
            return

        # if not compatibile with any species or self.species == []
        # create a new species with the given genome
        self.species.append(NEATSpecies(genome))

    def evolve(self):
        """
            Requires you to apply simulation(fitnes calculation) before calling...
            Applies selection, crossover, mutation on the current generation
            and evolves the population to the next generation...
            selection & crossover --> mutation --> evolution
        """
        # update the top genome across all generations
        # get the top genome in the current generation

        top = self.get_fittest_genome()
        if ((top.fitness-self.top_genome.fitness)>=self.fitenss_epsilon):
           self.top_genome = NEATGenome.Copy(top)
           self.top_genome_generation = self.current_generation

        # calculate the total adjusted fitnesses of each species
        total_fitness = sum(s.adjusted_fitnesses_sum for s in self.species)

        next_generation = []
        # -------------------------------------------------------------------#
        # -------------------------Mutation Only-----------------------------#
        mutate_only = int(self.mutate_only_chance * self.population_size)
        next_generation += [self.get_random_genome(copy=True) for _ in range(mutate_only)]
        # -------------------------------------------------------------------#

        # -------------------------------------------------------------------#
        # --------------------------Normal Mating----------------------------#
        # how many offsprings are left to complete the next generation
        left_offsprings = self.population_size-len(next_generation)

        for species in self.species:
            offspring_percentage = species.adjusted_fitnesses_sum/total_fitness
            offspring_count = int(offspring_percentage*left_offsprings)
            for _ in range(offspring_count):
                interspecies_crossover = (self.interspecies_mating_chance > random.uniform(0,1))
                parent1 = species.get_random_member()
                parent2 = self.get_random_genome() if interspecies_crossover else \
                          species.get_random_member()

                child = NEATGenome.Crossover(parent1, parent2)
                next_generation.append(child)

        # -------------------------------------------------------------------#
        # fill the rest (if any) with mutations of the top genome
        # the mutations will be applied later..
        left_offsprings = self.population_size-len(next_generation)
        next_generation += [NEATGenome.Copy(self.top_genome) for _ in range(left_offsprings)]
        # -------------------------------------------------------------------#

        for species in self.species:
            species.discard_members()

        for genome in next_generation:
            # apply mutation for each genome
            genome.mutate()

            # reset fitness as copying a genome keeps the fitness
            genome.fitness = 0
            genome.fitness_updated = False

            # add to the population
            self.add_genome(genome)

        # filter out the extinct species (empty species)
        self.species = [species for species in self.species if species.members]
        self.current_generation += 1

#
