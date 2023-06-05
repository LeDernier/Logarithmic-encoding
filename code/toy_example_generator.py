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

# plot the graph and save the figure
pos = nx.circular_layout(G)
nx.draw(G, pos, with_labels=True)
#nx.draw_networkx_edge_labels(G, pos)
path_fig = "../created_data/toy_graph_4.png"
plt.savefig(path_fig)
plt.clf()

# save the weighted edgelist of the graph
path_graph = "../created_data/toy_graph_4.col"
nx.write_weighted_edgelist(G, path_graph)