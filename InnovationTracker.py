#!/usr/bin/env python3

class InnovationTracker:
    def __init__(self):
        self.currrent_node_id = 0
        self.currrent_innovation_number = 0

        self.conn_history = dict()
        self.node_history = dict()

    def reset(self):
        self.currrent_node_id = 0
        self.currrent_innovation_number = 0

        self.conn_history.clear()
        self.node_history.clear()

    def get_innovation_number(self, conn):
        """ Returns an innovation number for conn, and adds it to the history"""
        innv_number = self.conn_history.get(str(conn), None)

        if (innv_number == None):
            innv_number = self.currrent_innovation_number
            self.conn_history[str(conn)] = innv_number
            self.currrent_innovation_number += 1

        return innv_number

    def get_node_id(self, node):
        """ Returns a node id for the given node based on its node_hash"""
        node_id = self.node_history.get(node.node_hash, None)

        if (node_id == None):
            node_id = self.currrent_node_id
            self.node_history[node.node_hash] = node_id
            self.currrent_node_id += 1

        return node_id
