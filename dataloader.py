import io
import numpy as np
import pandas as pd
import networkx as nx
from six.moves import urllib
from scipy.sparse import coo_matrix


class GraphLoader(object):
    def __init__(self, dataset: str = "wikipedia"):
        assert dataset in [
            "wikipedia",
            "twitch",
            "github",
            "facebook",
            "lastfm",
            "deezer",
        ], "Wrong dataset."
        self.dataset = dataset
        self.base_url = "https://github.com/benedekrozemberczki/karateclub/raw/master/dataset/node_level/"

    def pandas_reader(self, bytes):
        tab = pd.read_csv(
            io.BytesIO(bytes), encoding="utf8", sep=",", dtype={"switch": np.int32}
        )
        return tab

    def dataset_reader(self, end):
        path = self.base_url + self.dataset + "/" + end
        data = urllib.request.urlopen(path).read()
        data = self.pandas_reader(data)
        return data

    def get_graph(self) -> nx.classes.graph.Graph:
        data = self.dataset_reader("edges.csv")
        graph = nx.convert_matrix.from_pandas_edgelist(data, "id_1", "id_2")
        return graph

    def get_features(self) -> coo_matrix:
        data = self.dataset_reader("features.csv")
        row = np.array(data["node_id"])
        col = np.array(data["feature_id"])
        values = np.array(data["value"])
        node_count = max(row) + 1
        feature_count = max(col) + 1
        shape = (node_count, feature_count)
        features = coo_matrix((values, (row, col)), shape=shape)
        return features

    def get_target(self) -> np.array:
        data = self.dataset_reader("target.csv")
        target = np.array(data["target"])
        return target