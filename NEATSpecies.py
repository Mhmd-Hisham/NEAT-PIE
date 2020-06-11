#!/usr/bin/env python3

import random

from NEATGenome import NEATGenome
from NEATConfiguration import NEATConfiguration as config

class NEATSpecies:
    # compatibility distance constants
    c1 = config.c1
    c2 = config.c2
    c3 = config.c3
    compatibility_threshold = config.compatibility_threshold

    def __init__(self, network: NEATGenome, copy: bool=False):
        self.members = []
        self.representative = None

        self.tournament_sample_percentage = 0.50
        self.adjusted_fitnesses_sum = 0

        self.add_member(network, copy=copy, representative=True)

    def add_member(self,
                     new_member: NEATGenome,
                     representative: bool=False,
                     copy: bool=False):
        """ Adds a member to the species... Doesn't check if it's compatibile or not.

            if representative is set to True, updates the species representative
            to be the new member.

            if copy is set to True, adds a copy of the new member.
        """
        if (copy): new_member = NEATGenome.Copy(new_member)
        self.members.append(new_member)

        if (representative):
            self.representative = new_member

    def discard_members(self):
        """ Sets the representative to be a random member.
            Then removes all members of the species.
        """
        self.representative = self.get_random_member()
        self.members.clear()
        self.adjusted_fitnesses_sum = 0

    def get_fittest_member(self, sample: list=[]) -> NEATGenome:
        """ Returns the most fit genome in the given species sample.
            if sample is an empty list.. uses self.members (the whole species)
        """
        return max(sample if sample else self.members,
                   key=lambda genome: genome.fitness)

    def get_random_member(self, copy: bool=False) -> NEATGenome:
        """ Returns a random member from the species..
            Uses tournament selection...
        """

        # calculate the sample size
        sample_size = max(int(self.tournament_sample_percentage*len(self.members)), 2)
        sample_size = min(len(self.members), sample_size)

        # get the sample
        sample = random.sample(self.members, sample_size)
        # print(f"Sample Size: {sample_size}") #DEBUG

        # return the fittest
        fittest = self.get_fittest_member(sample)
        if (copy): fittest = NEATGenome.Copy(fittest)
        return fittest

    def is_compatible(self, other: NEATGenome):
        """ Tests whether the given Genonme belongs to the species..
            Returns True if it's compatibile, False otherwise..

            Computes the compatibility distance between the two genomes..
            returns true if it's less than the threshold, false otherwise.
        """
        # used for excess calcaultion
        min_innv = min(max(conn.innv for conn in self.representative.connections.values()),
                       max(conn.innv for conn in other.connections.values()))

        freq = {conn.innv:conn for conn in self.representative.connections.values()}

        excess = disjoint = matched = 0
        weight_difference = 0

        for conn in other.connections.values():
            gene = freq.get(conn.innv, None)
            if (gene != None):
                matched += 1
                weight_difference += abs(conn.weight-gene.weight)

            elif conn.innv<=min_innv:
                disjoint += 1

            else:
                excess += 1

        n = max(len(self.representative.connections),
                len(other.connections))

        n = 1.0 if (n<20) else float(n)

        # DEBUG
        # print(f"disjoint={disjoint}, excess={excess},  matched={matched}")

        compatibility_distance =  NEATSpecies.c1*(excess/n) + \
                                  NEATSpecies.c2*(disjoint/n)

        # just to avoid division by zero..
        compatibility_distance += 100 if matched==0 else \
                                  NEATSpecies.c3*(weight_difference/matched)

        return compatibility_distance <= NEATSpecies.compatibility_threshold

    def update_members_adjusted_fitness(self, *args, **kwargs) -> None:
        """ Calls the fitness function and calculates and updates
            the adjusted(shared) fitness for each member genome of the species.
        """

        for genome in self.members:
            genome.fitness = genome.fitness_function(genome, *args, **kwargs)
            genome.adjusted_fitness = genome.fitness/len(self.members)
            self.adjusted_fitnesses_sum += genome.adjusted_fitness
            genome.fitness_updated = True
#
