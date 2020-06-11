#!/usr/bin/env python3

import collections
import itertools
import math
from pprint import pprint
import random
from typing import Callable

from ConnectionGene import ConnectionGene
from InnovationTracker import InnovationTracker
from NEATConfiguration import NEATConfiguration as config
from NodeGene import Node, NodeGene

class NEATGenome:
    """ The NEATGenome class that represents a solution.
    """
    # check the NEATConfiguration class for the use of these values
    precision                = config.precision
    new_node_chance          = config.new_node_chance
    new_connection_chance    = config.new_connection_chance
    weight_perturb_chance    = config.weight_perturb_chance
    weight_mutation_chance   = config.weight_mutation_chance
    disable_inherited_chance = config.disable_inherited_chance

    def __init__(self,
                 nodes: list,
                 connections: list,
                 fitness_function: Callable[..., float],
                 innv_tracker: InnovationTracker,
                 fitness: float=0,
                 adjusted_fitness: float=0,
                 fitness_updated: bool=False):

        self.nodes = dict()
        self.connections = dict()

        self.innv_tracker = innv_tracker

         # node id :[list of incoming nodes]
        self.reverse_graph = collections.defaultdict(list)

        self.fitness = round(fitness, NEATGenome.precision)
        self.adjusted_fitness = round(adjusted_fitness, NEATGenome.precision)
        self.fitness_updated = fitness_updated

        # length of each layer basically
        # bias node doesn't count in the first layer..
        self.shape = collections.defaultdict(int)
        self.bias = NodeGene(Node.Bias, 0, 0, '0')
        self.add_node(self.bias)

        # you should always add connections before adding nodes
        # whenever you add a hidden node or an output node,
        # the add_node method tries to connect it to the bias node
        # and add_connection method ignores duplicate connections
        # so if the node is already connected to the bias,
        # you will lose the connection weight
        for conn in connections:
            self.add_connection(conn, copy=True)

        for node in nodes:
            self.add_node(node, copy=True)

        self.fitness_function = fitness_function

    @staticmethod
    def Copy(neat_genome):
        return NEATGenome(neat_genome.nodes.values(),
                          neat_genome.connections.values(),
                          neat_genome.fitness_function,
                          neat_genome.innv_tracker,
                          neat_genome.fitness,
                          neat_genome.adjusted_fitness,
                          neat_genome.fitness_updated)

    @staticmethod
    def Crossover(parent1, parent2):
        if ((parent1.fitness_updated == False) or
            (parent2.fitness_updated == False)):
           raise RuntimeError("Calculate fitness before crossing over!")

        # set parent2 to always be the more fit parent
        if (parent1.fitness > parent2.fitness):
            parent1, parent2 = parent2, parent1

        # since we inherit disjoint & excess genes from the more fit (parent2)
        # the topology of the child network will be the same as parent2
        # we can safely copy all nodes from parent2
        # don't inherit the bias node, as the constructor will create a new one
        child_nodes = [NodeGene.Copy(node) for node in parent2.nodes.values()
                       if (node.type != Node.Bias)]
        child_connections = []

        # map innovation numbers to connections
        freq = {conn.innv:conn for conn in parent1.connections.values()}
        for conn in parent2.connections.values():
            gene = freq.get(conn.innv, None)

            if (gene != None): # Matched
                gene = ConnectionGene.Copy(random.choice((gene, conn)))
                gene.disabled = (conn.disabled|gene.disabled)

                gene.disabled &= (random.uniform(0, 1) <
                                  NEATGenome.disable_inherited_chance)

            else: # Excess or Disjoint
                # inherit from the more fit (paretn2)
                gene = ConnectionGene.Copy(conn)

            child_connections.append(gene)

        return NEATGenome(child_nodes,
                          child_connections,
                          parent2.fitness_function,
                          parent2.innv_tracker)

    @staticmethod
    def BasicNetwork(layers: tuple,
                     fitness_function: Callable[..., float],
                     connections: list=[],
                     innv_tracker: InnovationTracker=None):
        """ Constructs a basic neural network with the given connections.
            node ids are numbered from 0..

            layers   -> a tuple representing the structure of the neural network.

            ex.
                nn = NEATGenome.BasicNetwork((2,4,3))
                nn -> a basic neural network with:
                        an input layer of 2 input nodes and a bias node,
                        a hidden layer with 4 hidden nodes, and
                        an output layer with 3 output nodes.
                adds the given connections to the network..
            raises TypeError if len(layers) < 2
        """
        if (len(layers)<2):
            raise TypeError("Can't create a neural network with one layer!!")

        # add the bias node...
        nodes = []
        next_id = 1

        for layerid, layer_size in enumerate(layers):
            node_type = Node.Sensor if layerid == 0 else \
                        Node.Hidden if layerid < len(layers)-1 else \
                        Node.Output

            for _ in range(layer_size):
                nodes.append(NodeGene(node_type, next_id, layerid, str(next_id)))
                next_id += 1

        return NEATGenome(nodes,
                          list(connections),
                          fitness_function,
                          innv_tracker)

    @staticmethod
    def FullFeedForwardNetwork(layers: tuple,
                               fitness_function: Callable[... , float],
                               innv_tracker: InnovationTracker=None,
                               default_weight: float=None):
        """ Returns a full feedforward neural network with nodes and connections initialized..
            Connection weights are randomized..

            layers -> a tuple representing the structure of the neural network.
            default_weight -> resets the weights to the given value if not None

            ex.
                nn = NEATGenome.FullFeedForwardNetwork((2,4,3), 1.0)
                nn -> a fully connected neural network with:
                        an input layer of 2 input nodes and a bias node,
                        a hidden layer with 4 hidden nodes, and
                        an output layer with 3 output nodes.
                      with all weights set to 1.0.

            # Bias node connections are always set to 0.0
            # Bias node id is always 0
            # Bias layerid is always 0

            return a NEATGenome object
        """
        # connections = [ConnectionGene(0, outgoing, 0.0)
        #                for outgoing in range(layers[0]+1, sum(layers)+1)]
        connections = []
        id_shift = 1

        # don't process the last layer
        for index, layer_size in enumerate(layers[:-1]):
            for incoming in range(layer_size):
                for outgoing in range(layers[index+1]):
                    connections.append(ConnectionGene(incoming+id_shift,
                                                      outgoing+id_shift+layer_size,
                                                      default_weight))
                    # print(incoming+id_shift, end=", ") #DEBUG
                    # print(outgoing+id_shift+layer_size) #DEBUG
            id_shift += layer_size

        return NEATGenome.BasicNetwork(layers,
                                       fitness_function,
                                       connections,
                                       innv_tracker)

    def add_connection(self,
                       conn: ConnectionGene,
                       copy: bool=False):
        """
            Adds the given connection to the phenotype of the genome.
            Returns True if the connection was added successfully,
                    False if it's a duplicate connection.
        """
        if copy: conn = ConnectionGene.Copy(conn)
        if (str(conn) in self.connections): return False

        conn.innv = self.innv_tracker.get_innovation_number(conn)
        self.connections[str(conn)] = conn
        self.reverse_graph[conn.outgoing].append(conn)
        return True

    def add_node(self,
                 node: NodeGene,
                 copy: bool=False,
                 new_layer: bool=False):
        """
            Adds the given node to the phenotype of the genome.
            if new_layer is set to true, inserts the node in a new layer.

            Returns True if the node was added successfully,
                    False if it's a duplicate node.
        """
        if copy: node = NodeGene.Copy(node)

        node.id = self.innv_tracker.get_node_id(node)

        if (node.id in self.nodes): return False

        if (new_layer):
            for nn_node in self.nodes.values():
                nn_node.layerid += (nn_node.layerid>=node.layerid)

        # don't count the bias node in it's layer
        self.nodes[node.id] = node
        self.shape[node.layerid] += (node.type!=Node.Bias)

        # connect to the bias node
        if (node.type == Node.Hidden) or (node.type == Node.Output):
            self.add_connection(ConnectionGene(self.bias.id, node.id, 0.0))

        return True

    def predict(self, nn_input: dict):
        if (len(nn_input) != self.shape[0]):
            raise RuntimeError(f"Expected {self.shape[0]} inputs, got {len(nn_input)}..")

        visited = {node_id:(False, 0.0) for node_id in self.nodes}

        def evaluate(node_id):
            node = self.nodes[node_id]
            if (node.type == Node.Bias): return 1.0
            if (node.type == Node.Sensor): return nn_input[node.id]
            if (visited[node.id][0]): return visited[node.id][1]

            # calculate the weighted sum
            connections = (conn for conn in self.reverse_graph[node.id]
                           if conn.disabled == False)

            ws = sum(evaluate(conn.incoming)*conn.weight for conn in connections)

            visited[node.id] = (True, round(self.activation_func(ws),
                                            NEATGenome.precision))

            return visited[node_id][1]

        return {node_id: evaluate(node_id)
                for node_id in self.nodes
                if (self.nodes[node_id].type == Node.Output)}

    def activation_func(self, gamma: float):
        # https://stackoverflow.com/questions/36268077/overflow-math-range-error-for-log-or-exp
        # return 1/(1+math.exp(-4.9*gamma)) # steepened sigmoid
        if gamma < 0:
            return 1 - 1 / (1 + math.exp(gamma*4.9))

        return 1 / (1 + math.exp(-4.9*gamma))

    def connection_mutation(self):
        """ Tries to mutate the network to add a new connection
            returns True if the network is successfully mutated...
        """
        layers = len(self.shape)
        layer1 = random.randrange(0, layers-1)
        layer2 = random.randrange(layer1+1, layers)

        #DEBUG
        if (layer1>=layer2):
            raise ValueError(f"Wrong choice of layers.. {layer1}, {layer2}")

        # now that we have two layers, let's get the nodes in each layer
        layer1 = [node for node in self.nodes.values() if node.layerid==layer1]
        layer2 = [node for node in self.nodes.values() if node.layerid==layer2]

        return self.add_connection(ConnectionGene(random.choice(layer1).id,
                                                  random.choice(layer2).id))

    def node_mutation(self):
        """ Tries to mutate the network to add a new node..
            returns a node gene and two connection genes without adding
            anything to the network..
        """
        # list of active connections
        # exclude the bias connections..
        active_connections = [conn for conn in self.connections.values()
                               if (conn.disabled == False) and
                                  (conn.incoming != self.bias.id)]


        # No active connections to add nodes between
        if (not active_connections): return False

        conn = random.choice(active_connections)

        node = NodeGene(Node.Hidden,
                        -1,
                        self.nodes[conn.incoming].layerid+1,
                        str(conn))

        # try to add the node
        # ** node.id will chance if it's added to the network **
        is_in_new_layer = (node.layerid == self.nodes[conn.outgoing].layerid)
        node_added = self.add_node(node, new_layer=is_in_new_layer)

        if (node_added == False): return False
        # if the node was successfully added
        # disable the connection and put the new connections
        conn.disabled = True
        self.add_connection(ConnectionGene(conn.incoming, node.id))
        self.add_connection(ConnectionGene(node.id, conn.outgoing))

        return True

    def mutate(self):
        # a mutation might change the topology/weights of the network
        # so the fitness must be recalculated for the new network
        self.fitness_updated = False

        # ------------------------------------------------------------------- #
        # ----------------------Connection Mutation-------------------------- #
        if (NEATGenome.new_connection_chance > random.uniform(0, 1)):
            self.connection_mutation()

        if (NEATGenome.weight_mutation_chance > random.uniform(0, 1)):
            for conn in self.connections.values():
                # ignore disabled connections
                if conn.disabled: continue

                randomize = NEATGenome.weight_perturb_chance < random.uniform(0, 1)
                conn.perturb_weight(randomize=randomize)

                # normalize the weight if it's a bias connection
                if (conn.incoming == self.bias.id):
                    conn.normalize_weight()

        # ------------------------------------------------------------------- #
        # --------------------------Node Mutation---------------------------- #
        if (NEATGenome.new_node_chance > random.uniform(0, 1)):
            self.node_mutation()
        # ------------------------------------------------------------------- #

        return

#
