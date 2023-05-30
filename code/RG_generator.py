# Imports non Qiskit
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# custom modules
from graph_functions import get_laplacian_matrix

### 1. GENERATE RANDOM INSTANCE ###

def generate_random_graphs(list_num_vars: list, num_inst: int=1, 
                           weight_min: float= 0, weight_max: float=1, 
                            seed=123, suffix_name=""):
    """
        Parameters:
            - list_num_vars: list with the number of nodes (to benchmark the effect of the size)
            - num_inst: number of instance (by number of nodes)
            - weight_min/weight_max: minimum and maximum weight
            - seed: random seed
            - suffix_name: string added as a prefix to all the generated files
    """
    

    random.seed(seed)
    num_generated_graphs = 0
    for num_nodes in list_num_vars:
        log2_numvars = int(np.log2(num_nodes))      # number of qubits in log-encoding
        #print("log2_numvars: ", log2_numvars)
        for inst in range(num_inst):

            G = nx.gnm_random_graph(num_nodes,seed)   # unweighted graph
            for (u,v) in G.edges():
                G.edges[u,v]["weight"] = random.randint(weight_min,weight_max)

            ## save the figure of the graph ##

            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True)
            nx.draw_networkx_edge_labels(G, pos)
            path_fig = "../created_data/random_graphs/rand_graph_"+str(num_nodes)+"."+str(inst)+suffix_name +".png"
            plt.savefig(path_fig)
            plt.clf()
            #plt.show()

            path_graph = "../created_data/random_graphs/"+suffix_name+"rand_graph_"+str(num_nodes)+"."+str(inst)+suffix_name+".col"
            nx.write_weighted_edgelist(G, path_graph)
            #print()

            ## add the objective function ##

            #laplacian = get_laplacian_matrix(G)             # Laplacian matrix
            #qubo = (-num_nodes/4)*laplacian                  # QUBO matrix
            #print("QUBO matrix: \n", qubo)
            num_generated_graphs += 1

    return num_generated_graphs

 
if __name__ == '__main__':
    # quick test of the random generator

    seed = 123
    list_num_nodes = [4,8,16,32]               # number of variables/nodes
    w_min = -5          # minimum weight    
    w_max = 5          # maximum weight
    num_inst = 1        # number of instances to test
    suffix_name = ""


    num_RG = generate_random_graphs(list_num_nodes, num_inst, w_min, w_max, seed, suffix_name)
    print("number of random graphs generated: ", num_RG)

    #print("log2_numvars: ", log2_numvars)