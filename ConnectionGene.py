#!/usr/bin/env python3

import random

from NEATConfiguration import NEATConfiguration as config
from NodeGene import Node, NodeGene

class ConnectionGene:
    """ NEAT's abstract neural network connection gene """
    precision  = config.precision
    min_weight = config.minimum_weight
    max_weight = config.maximum_weight

    def __init__(self,
                 in_node: int,
                 out_node: int,
                 weight: float=None,
                 innv: int=None,
                 disabled: bool=False):
        self.incoming = in_node
        self.outgoing = out_node

        self.weight   = weight
        self.disabled = disabled
        self.innv     = innv         # inovation number(historical mark)

        if (weight == None):
            self.perturb_weight(randomize=True)

    @staticmethod
    def Copy(conn):
        return ConnectionGene(conn.incoming,
                              conn.outgoing,
                              conn.weight,
                              conn.innv,
                              conn.disabled,)

    def normalize_weight(self, min: int=-1.0, max: int=1.0):
        """ normalizes the weight of the connection between min and max"""
        self.weight = max if (self.weight > max) else \
                      min if (self.weight < min) else \
                      self.weight

    def perturb_weight(self, randomize:bool=False):
        """ Perturb the weight of the connection.
            if randomize is set to True, the weight is completely randomized.
        """
        if (randomize):
            self.weight = random.uniform(ConnectionGene.min_weight,
                                         ConnectionGene.max_weight)

        else:
            self.weight += random.uniform(ConnectionGene.min_weight/2,
                                          ConnectionGene.max_weight/2)

        # normalize the weight
        self.normalize_weight(ConnectionGene.min_weight,
                              ConnectionGene.max_weight)

        self.weight = round(self.weight, ConnectionGene.precision)

    def __repr__(self):
        """ For debugging purposes.. use pprint.pprint to print the connection"""
        return f"ConnectionGene({self.incoming}, " + \
               f"{self.outgoing}, " + \
               f"{self.weight}, " + \
               f"{self.innv}, " + \
               f"{self.disabled})"

    def __str__(self):
        """ For hashing purposes.. """
        return f"{self.incoming:07.0f}:{self.outgoing:07.0f}"
