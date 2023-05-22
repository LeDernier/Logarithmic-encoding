# Imports non Qiskit
import random
import sys
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import networkx as nx
# optimizers
from scipy.optimize import minimize

### AUXILIARY FUNCTIONS ###

def get_weight_matrix(G):
    """
    :param G: nx.Graph: a fully connected graph with the random assigned weights
    :return: numpy array: a matrix containing weights of edges
    """
    matrix = []
    for i in range(len(G)):
        row = []
        for j in range(len(G)):
            if j != i:
                row.append(G._adj[i][j]['weight'])
            else:
                row.append(0)
        matrix.append(row)
    return np.array(matrix)

def get_adjacency_matrix(G):
    return np.array(nx.adjacency_matrix(G).todense())

def get_laplacian_matrix(G):
    return np.array(nx.laplacian_matrix(G).todense())


### 1. GENERATE RANDOM INSTANCE ###
seed = 123
num_vars = 4               # number of variables/nodes
log2_numvars = int(np.log2(num_vars))      # number of qubits in log-encoding
w_min = 1           # minimum weight    
w_max = 10          # maximum weight
num_inst = 1        # number of instances to test
print("log2_numvars: ", log2_numvars)

random.seed(seed)
for inst in range(num_inst):

    G = g=nx.gnm_random_graph(num_vars,seed)   # unweighted graph
    for (u,v) in G.edges():
        G.edges[u,v]["weight"] = random.randint(w_min,w_max)

    ## save the figure of the graph ##

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos)
    path_fig = "..\\created_data\\random_graphs\\rand_graph_" + str(inst) + ".png"
    plt.savefig(path_fig)
    #plt.show()

    path_graph = "..\\created_data\\random_graphs\\rand_graph_"+str(num_vars)+"."+str(inst)
    print(nx.write_weighted_edgelist(G, path_graph))

    """ ## save the graph instance in ".col" format ##
    data_graph = open("") """

    ## add the objective function ##

    laplacian = get_laplacian_matrix(G)             # Laplacian matrix
    qubo = (-num_vars/4)*laplacian                  # QUBO matrix
    print("QUBO matrix: \n", qubo)

 
