import hashlib
import networkx as nx
from typing import List, Dict


class WeisfeilerLehmanHashing(object):
    """
    Weisfeiler-Lehman feature extractor class.
    """

    def __init__(
        self,
        graph: nx.classes.graph.Graph,
        wl_iterations: int,
        attributed: bool,
        erase_base_features: bool,
    ):

        self.wl_iterations = wl_iterations
        self.graph = graph
        self.attributed = attributed
        self.erase_base_features = erase_base_features
        self._set_features()
        self._do_recursions()

    def _set_features(self):
        if self.attributed:
            self.features = nx.get_node_attributes(self.graph, "feature")
        else:
            self.features = {
                node: self.graph.degree(node) for node in self.graph.nodes()
            }
        self.extracted_features = {k: [str(v)] for k, v in self.features.items()}

    def _erase_base_features(self):

        for k, v in self.extracted_features.items():
            del self.extracted_features[k][0]

    def _do_a_recursion(self):
        new_features = {}
        for node in self.graph.nodes():
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])] + sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = {
            k: self.extracted_features[k] + [v] for k, v in new_features.items()
        }
        return new_features

    def _do_recursions(self):
        for _ in range(self.wl_iterations):
            self.features = self._do_a_recursion()
        if self.erase_base_features:
            self._erase_base_features()

    def get_node_features(self) -> Dict[int, List[str]]:
        return self.extracted_features

    def get_graph_features(self) -> List[str]:
        return [
            feature
            for node, features in self.extracted_features.items()
            for feature in features
        ]