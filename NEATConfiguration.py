#!/usr/bin/env python3

class NEATConfiguration:
    population_size = 150
    generations = 100

    # ------------------------- Configuration ----------------------------#
    # used for rounding all the weights/fitnesses
    precision = 6

    # min and max weight for any connection
    # the algorithm still works if you disable weight normalization
    # but you might see an explosion in the number of species.
    # bias connections are always normalized from -1.0 to +1.0
    # you can change this in the ConnectionGene class
    minimum_weight = -4
    maximum_weight = +4

    # uses for comparing two fitess values
    fitenss_epsilon = 2e-2
    # ------------------------- Mutation Rates ---------------------------#
    # chance for mutating a gene's weights
    weight_mutation_chance = 0.80

    # otherwise the weight is assigned a new random value
    weight_perturb_chance = 0.90

    # new connection/node chances
    new_node_chance = 0.03
    new_connection_chance = 0.3
    # --------------------------------------------------------------------#
    # ------------------------- Crossover Rates --------------------------#

    # chance of disabling an inherited gene if it is disabled in either parent
    disable_inherited_chance = 0.75

    # The chance for a geneome to only mutate without crossing over
    # while still evolving to the next generation
    mutate_only_chance = 0.25

    # interspecies mating chance
    interspecies_mating_chance = 0.001
    # --------------------------------------------------------------------#
    # -------------------Compatibility Distance Parameters----------------#

    # The coefficients for measuring compatibility distance
    c1 = 1.0
    c2 = 1.0
    c3 = 3.0

    compatibility_threshold = 3.0
    # --------------------------------------------------------------------#
