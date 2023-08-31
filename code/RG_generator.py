# Imports non Qiskit
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# custom modules
from graph_functions import get_laplacian_matrix

### 1. GENERATE RANDOM INSTANCE ###

def generate_w_random_graphs(list_num_nodes: list, num_inst: int=1, 
                           weight_min: float= 0, weight_max: float=1, 
                            seed=123, suffix_name=""):
    """
        It generates a random instance of Weighted Max-Cut (the QUBO matrix).

        Parameters:
            - list_num_nodes: list with the number of nodes (to benchmark the effect of the size)
            - num_inst: number of instance (by number of nodes)
            - weight_min/weight_max: minimum and maximum weight
            - seed: random seed
            - suffix_name: string added as a prefix to all the generated files
    """
    

    random.seed(seed)
    num_generated_graphs = 0
    for num_nodes in list_num_nodes:
        for inst in range(1,num_inst+1):

            G = nx.gnm_random_graph(num_nodes,seed)   # unweighted graph
            for (u,v) in G.edges():
                G.edges[u,v]["weight"] = random.randint(weight_min,weight_max)

            ## save the figure of the graph ##

            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True)
            try:
                nx.draw_networkx_edge_labels(G, pos)
            except:
                continue

            path_fig = "created_data/random_graphs/wrand_graph_"+str(num_nodes)+"."+str(inst)+suffix_name +".png"
            plt.savefig(path_fig)
            plt.clf()
            #plt.show()

            path_graph = "created_data/wrandom_graphs/"+suffix_name+"rand_graph_"+str(num_nodes)+"."+str(inst)+suffix_name+".col"
            nx.write_weighted_edgelist(G, path_graph)

            num_generated_graphs += 1

    return num_generated_graphs


def generate_random_graphs(list_num_nodes: list, list_densities: list, num_inst: int=1, 
                            seed=123, suffix_name=""):
    """
        It generates a random instance of Weighted Max-Cut (the QUBO matrix).

        Parameters:
            - list_num_nodes: list with the number of nodes (to benchmark the effect of the size)
            - list_densities: list with the desired densities (number of edges/n^2)
            - num_inst: number of instance (by number of nodes)
            - seed: random seed
            - suffix_name: string added as a prefix to all the generated files
    """
    

    random.seed(seed)
    num_generated_graphs = 0
    for num_nodes in list_num_nodes:
        for density in list_densities:
            for inst in range(1,num_inst+1):

                G = nx.erdos_renyi_graph(num_nodes, p=density)   # unweighted graph
                #n_edges = int(density*num_nodes*(num_nodes-1))
                #G = nx.gnm_random_graph(num_nodes, n_edges)

                """ ## save the figure of the graph ##

                pos = nx.spring_layout(G)
                nx.draw(G, pos, with_labels=True)
            
                path_fig = "created_data/random_graphs/rand_graph_N"+str(num_nodes) +\
                      "-D" + str(density) +"-I"+str(inst)+suffix_name +".png"
                plt.savefig(path_fig)
                plt.clf()
                #plt.show() """

                path_graph = "created_data/random_graphs/"+suffix_name+"rand_graph_N"+\
                    str(num_nodes)+ "-D" + str(density)+"-I"+str(inst)+suffix_name+".col"       # N:nodes, D:density, I:instance
                nx.write_edgelist(G, path_graph)

                num_generated_graphs += 1

    return num_generated_graphs


def generate_star_graphs(list_num_nodes: list, num_inst: int=1,
                            seed=123, suffix_name=""):
    """
        Parameters:
            - list_num_nodes: list with the number of nodes (to benchmark the effect of the size)
            - num_inst: number of instance (by number of nodes)
            - seed: random seed
            - suffix_name: string added as a prefix to all the generated files
    """
    

    random.seed(seed)
    num_generated_graphs = 0
    for num_nodes in list_num_nodes:
        for inst in range(num_inst):

            G = nx.Graph()
            G.add_nodes_from(list(range(1,num_nodes + 1)))
            for v in range(2,num_nodes + 1):
                G.add_edge(1,v)

            ## save the figure of the graph ##

            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True)
            #nx.draw_networkx_edge_labels(G, pos)
            path_fig = "../created_data/star_graphs/star_graph_"+str(num_nodes)+"."+str(inst)+suffix_name +".png"
            plt.savefig(path_fig)
            plt.clf()
            #plt.show()

            path_graph = "../created_data/star_graphs/"+suffix_name+"star_graph_"+str(num_nodes)+"."+str(inst)+suffix_name+".col"
            nx.write_edgelist(G, path_graph)

            ## add the objective function ##

            #laplacian = get_laplacian_matrix(G)             # Laplacian matrix
            #qubo = (-num_nodes/4)*laplacian                  # QUBO matrix
            #print("QUBO matrix: \n", qubo)
            num_generated_graphs += 1

    return num_generated_graphs


def test_wrand_graphs():
    # quick test of the random generator for weighted graphs

    seed = 123
    list_num_nodes = [4,8,16,32,64]               # number of variables/nodes
    w_min = 0          # minimum weight    
    w_max = 1          # maximum weight
    num_inst = 10        # number of instances to test
    suffix_name = ""


    num_RG = generate_random_graphs(list_num_nodes, num_inst, w_min, w_max, seed, suffix_name)
    print("number of random graphs generated: ", num_RG)


def test_rand_graphs():
    # quick test of the random generator

    seed = 123
    #list_num_nodes = [4,8,12,16,20,24,32,36,48,64]               # number of variables/nodes
    #list_num_nodes = [72,96]               # number of variables/nodes
    #list_num_nodes = [56,84]
    list_num_nodes = [104,116,128,192,256]
    list_densities = (np.array(list(range(1,10,2)))/10).tolist()  
    num_inst = 10        # number of instances to test
    suffix_name = ""


    num_RG = generate_random_graphs(list_num_nodes, list_densities, num_inst, seed, suffix_name)
    print("number of random graphs generated: ", num_RG)

def test_star_graphs():
     # quick test of the random generator

    seed = 123
    list_num_nodes = [2,4,8,12,16,20,24,28,32]               # number of variables/nodes
    num_inst = 5        # number of instances to test
    suffix_name = ""


    num_RG = generate_star_graphs(list_num_nodes, num_inst, seed, suffix_name)
    print("number of random graphs generated: ", num_RG)
 
if __name__ == '__main__':
    test_rand_graphs()
    #test_star_graphs()

    