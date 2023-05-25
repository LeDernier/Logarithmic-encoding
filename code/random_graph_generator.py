# Imports non Qiskit
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# custom modules
from graph_functions import get_laplacian_matrix

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

 
