import random
import numpy as np
import networkx as nx
from typing import List
import re


class Base(object):
    seed: int

    def __init__(self):
        pass

    def train(self):
        pass

    def get_embedding(self):
        pass

    def get_memberships(self):
        pass

    def get_cluster_centers(self):
        pass
    
    def get_params(self):
        rx = re.compile(r'^\_')
        params = self.__dict__
        params = {key: params[key] for key in params if not rx.search(key)}
        return params

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    @staticmethod
    def ensure_integrity(graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
        edge_list = [(index, index) for index in range(graph.number_of_nodes())]
        graph.add_edges_from(edge_list)

        return graph

    @staticmethod
    def check_indexing(graph: nx.classes.graph.Graph):
        numeric_indices = [index for index in range(graph.number_of_nodes())]
        node_indices = sorted([node for node in graph.nodes()])

        assert numeric_indices == node_indices, "The node indexing is wrong."

    def check_graph(self, graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
        self.check_indexing(graph)
        graph = self.ensure_integrity(graph)

        return graph

    def check_graphs(self, graphs: List[nx.classes.graph.Graph]):
        graphs = [self.check_graph(graph) for graph in graphs]

        return graphs