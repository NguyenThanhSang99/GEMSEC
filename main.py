import networkx as nx
from model import GEMSEC
from dataloader import GraphLoader

reader = GraphLoader()

graph = reader.get_graph()

model = GEMSEC()

model.train(graph)

print(model.get_embedding())
