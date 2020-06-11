#!/usr/bin/env python3

import enum

class Node(enum.Enum):
    """ An Enumeration class for Node Types """
    Sensor = 0
    Hidden = 1
    Output = 2
    Bias   = 3

class NodeGene:
    """ NEAT's abstract neural network node gene """
    def __init__(self,
                 node_type: Node,
                 node_id: int,
                 layerid: int,
                 node_hash: str=""):

        self.type = node_type
        self.id   = node_id
        self.layerid = layerid

        # used for id tracking for hidden nodes only
        self.node_hash = node_hash

    def __repr__(self):
        return f"NodeGene({self.type}, " + \
               f"{self.id}, " + \
               f"{self.layerid}, " + \
               f"'{self.node_hash}')"

    @staticmethod
    def Copy(node_gene):
        return NodeGene(node_gene.type,
                        node_gene.id,
                        node_gene.layerid,
                        node_gene.node_hash)
