import random
import numpy as np
import networkx as nx
from typing import Dict
from base import Base
from randomwalk import RandomWalk


class GEMSEC(Base):
    def __init__(
        self,
        walk_number: int = 5,
        walk_length: int = 80,
        dimensions: int = 32,
        negative_samples: int = 5,
        window_size: int = 5,
        learning_rate: float = 0.1,
        clusters: int = 10,
        gamma: float = 0.1,
        seed: int = 42,
    ):
        self.walk_number = walk_number
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.negative_samples = negative_samples
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.clusters = clusters
        self.gamma = gamma
        self.seed = seed

    def setup_sampling_weights(self, graph):
        self._sampler = {}
        index = 0
        for node in graph.nodes():
            for _ in range(graph.degree(node)):
                self._sampler[index] = node
                index = index + 1
        self._global_index = index - 1

    def initialize_node_embeddings(self, graph):
        shape = (graph.number_of_nodes(), self.dimensions)
        self._base_embedding = np.random.normal(0, 1.0 / self.dimensions, shape)

    def initialize_cluster_centers(self, graph):
        shape = (self.dimensions, self.clusters)
        self._cluster_centers = np.random.normal(0, 1.0 / self.dimensions, shape)

    def sample_negative_samples(self):
        negative_samples = [
            self._sampler[random.randint(0, self._global_index)]
            for _ in range(self.negative_samples)
        ]
        return negative_samples

    def calculcate_noise_vector(self, negative_samples, source_node):
        noise_vectors = self._base_embedding[negative_samples, :]
        source_vector = self._base_embedding[int(source_node), :]
        raw_scores = noise_vectors.dot(source_vector.T)
        raw_scores = np.exp(np.clip(raw_scores, -15, 15))
        scores = raw_scores / np.sum(raw_scores)
        scores = scores.reshape(-1, 1)
        noise_vector = np.sum(scores * noise_vectors, axis=0)
        return noise_vector

    def calculate_cluster_vector(self, source_node):
        distances = (
            self._base_embedding[int(source_node), :].reshape(-1, 1)
            - self._cluster_centers
        )
        scores = np.power(np.sum(np.power(distances, 2), axis=0), 0.5)
        cluster_index = np.argmin(scores)
        cluster_vector = distances[:, cluster_index] / scores[cluster_index]

        return cluster_vector, cluster_index

    def do_descent_for_pair(self, negative_samples, source_node, target_node):
        noise_vector = self.calculcate_noise_vector(negative_samples, source_node)
        target_vector = self._base_embedding[int(target_node), :]
        cluster_vector, cluster_index = self.calculate_cluster_vector(source_node)
        node_gradient = noise_vector - target_vector + self.gamma * cluster_vector
        node_gradient = node_gradient / np.linalg.norm(node_gradient)
        self._base_embedding[int(source_node), :] += -self.learning_rate * node_gradient
        self._cluster_centers[:, cluster_index] += (
            self.learning_rate * self.gamma * cluster_vector
        )

    def update_a_weight(self, source_node, target_node):
        negative_samples = self.sample_negative_samples()
        self.do_descent_for_pair(negative_samples, source_node, target_node)
        self.do_descent_for_pair(negative_samples, target_node, source_node)

    def do_gradient_descent(self):
        random.shuffle(self._walker.walks)
        for walk in self._walker.walks:
            for i, source_node in enumerate(
                walk[: self.walk_length - self.window_size]
            ):
                for step in range(1, self.window_size + 1):
                    target_node = walk[i + step]
                    self.update_a_weight(source_node, target_node)

    def train(self, graph: nx.classes.graph.Graph):
        self.set_seed()
        graph = self.check_graph(graph)
        self.setup_sampling_weights(graph)
        self._walker = RandomWalk(self.walk_length, self.walk_number)
        self._walker.do_walks(graph)
        self.initialize_node_embeddings(graph)
        self.initialize_cluster_centers(graph)
        self.do_gradient_descent()

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.
        Return types:
            * **embedding** *(Numpy array)*: The embedding of nodes.
        """
        return np.array(self._base_embedding)

    def get_membership(self, node):
        distances = self._base_embedding[node, :].reshape(-1, 1) - self._cluster_centers
        scores = np.power(np.sum(np.power(distances, 2), axis=0), 0.5)
        cluster_index = np.argmin(scores)
        return cluster_index

    def get_memberships(self) -> Dict[int, int]:
        memberships = {
            node: self._get_membership(node)
            for node in range(self._base_embedding.shape[0])
        }
        return memberships