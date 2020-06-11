#!/usr/bin/env python3


from ConnectionGene import ConnectionGene
from NodeGene import Node
from NodeGene import NodeGene
from NEATGenome import NEATGenome
from graphviz import Digraph

from random import choice
from string import ascii_lowercase

random_filename = lambda n: ''.join(choice(ascii_lowercase) for _ in range(n))

class NNVisualizer:
    def __init__(self, nn: NEATGenome, filename=None):
        self.filename = filename if filename != None else random_filename(10)
        self.dot = Digraph(comment='Neural Network', filename=self.filename)
        self.dot.attr("node", shape='circle')
        self.dot.attr("graph", rankdir="LR")
        self.dot.format = 'png'
        self.nn = nn
        self.node_width = "0.4"
        self.node_height = "0.4"

        self.plot1()

    def plot1(self):
        for node in self.nn.nodes.values():
            self.dot.node(str(node.id),
                          color="dimgrey" if (node.type==Node.Hidden) else \
                                "crimson" if (node.type==Node.Bias) else \
                                "brown4"  if (node.type==Node.Output) else \
                                "blue")

        for source in self.nn.reverse_graph:
            for conn in self.nn.reverse_graph[source]:
                # if (conn.disabled): continue
                self.dot.edge(str(conn.incoming),
                              str(conn.outgoing),
                              # label=str(round(conn.weight, 3)),
                              color="pink" if self.nn.nodes[conn.incoming].type == Node.Bias else
                                    "red"  if (conn.disabled) else
                                    "limegreen")

    def plot2(self):
        number_of_bias_nodes = max(node.layerid for node in self.nn.nodes.values())
        for i in range(number_of_bias_nodes):
            self.dot.node(f"bias-{i}",
                          pos=f"{i},0!",
                          shape="point",
                          width=self.node_width,
                          height=self.node_height,
                          color="crimson")

        for node in self.nn.nodes.values():
            if node.type == Node.Bias: continue
            self.dot.node(str(node.id),
                          pos=f"{node.layerid},{node.id}!",
                          shape="point",
                          width=self.node_width,
                          height=self.node_height,
                          color="dimgrey" if (node.type==Node.Hidden) else \
                                "brown4"  if (node.type==Node.Output) else \
                                "blue")

        for source in self.nn.reverse_graph:
            for conn in self.nn.reverse_graph[source]:
                # if (conn.disabled): continue
                if conn.incoming == self.nn.bias.id:
                    node = self.nn.nodes[conn.outgoing]
                    self.dot.edge(f"bias-{node.layerid-1}",
                                  str(conn.outgoing),
                                  color="pink")
                    continue

                self.dot.edge(str(conn.incoming),
                              str(conn.outgoing),
                              # label=str(round(conn.weight, 3)),
                              color="red"  if (conn.disabled) else "limegreen")


    def view(self):
        self.dot.view()
