# Imports non Qiskit
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# custom modules
from graph_functions import get_laplacian_matrix

G = nx.Graph()
G.add_nodes_from([0,1,2,3])
G.add_weighted_edges_from([(0,1,1),(0,3,1),(1,2,1)])

path_graph = "../created_data/toy_graph_4.col"
nx.write_weighted_edgelist(G, path_graph)