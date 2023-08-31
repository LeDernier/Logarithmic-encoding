import sys
import os
from time import time, perf_counter

import pandas as pd
import numpy as np
import pulp as pl
import networkx as nx

## Imports customs
sys.path.append('./code')
from graph_functions import get_laplacian_matrix, get_adjacency_matrix
from expectation_functions import get_max_depth, R_map, Rf_map, Rfs_map


def test_unit(Hmat):
    return None


def unit_test_max_cut(graph, m, data_log, row_dict):
    """
        Resolution of the instance of max-cut exactly. This function modifies a dictionary and 
        a list that will be used to record the results.
    
        Parameters:
            - m: number of qubits.

        Returns:
            - Optimal value.
    """

    # parameters
    N = 2**m
    nodes = graph.nodes()
    edges = graph.edges()

    opt_val = 0
        
    if len(edges) > 0:
        
        # model
        prob = pl.LpProblem("Standard max-cut", pl.LpMaximize)

        # variables
        x = pl.LpVariable.dicts("Node", nodes, 0, cat=pl.LpBinary)     # decision variables (=1 if the node is in the first partition)
        z = pl.LpVariable.dicts("Edge", edges, 0, cat=pl.LpBinary)     # linearization variables

        # objective function (simple linearization)
        obj = pl.lpSum([x[edge[0]] + x[edge[1]] - 2*z[edge] for edge in edges])
        prob += obj

        ## constraints
        # linearization constraints
        for edge in edges:
            prob += (z[edge] <= x[edge[0]], "linear1_"+str(edge))
            prob += (z[edge] <= x[edge[1]], "linear2_"+str(edge))
            prob += (z[edge] >= x[edge[0]] + x[edge[1]] - 1, "linear3_"+str(edge))

        
        solver = pl.getSolver('CPLEX_CMD', msg = False)
        t_start_solver = perf_counter()
        status = prob.solve(solver)
        time_resolution = perf_counter() - t_start_solver
        opt_val = -obj.value()
    else:
        status = "Optimal Solution Found"
        time_resolution = 0.0
        opt_val = 0.0

    row_dict["status"] =  1 # number associated to the state "LpStatusOptimal"
    row_dict["time_resolution"] = time_resolution
    row_dict["opt_value"] = opt_val
    
   
    data_log.append(list(row_dict.values()))
    
    return opt_val




def run_tests_max_cut(list_n_vars: list, list_densities: list, max_instances = 10):
    """
        list_n_vars: list with the number of variables to test.
        list_densities: list with the edge densities to test.
        max_instances: maximum number of repetitions of the evaluation of a quantum circuit
    """

    ## Path
    # folder Path
    path = os.getcwd()
    #path = os.path.join(path,"..\\created_data\\random_graphs")
    #print("current working directory: ", path)
    
    # change the directory
    os.chdir(path)
 
    
    # iterate through all file
    root_name = ""

    # id_instance: i from 1 to test_max. It is the i-th repetition of the experiment (the only difference betwee two tests is the sampling of the quantum circuit)
    # max_circuit_depth: largest depth of all the circuits used (associated to the families of pairwise commuting Pauli strings)
    cols_data = ["n_vars", "density", "id_instance"]
    cols_values = ["status", "opt_value"]
    cols_depths = []
    cols_times = ["time_resolution"]
    columns = cols_data + cols_values + cols_depths + cols_times

    data_log = []      # mutable object to store the 
    row_dict = {columns[i]: None for i in range(len(columns))}

    # n_qubits: number of qubits
    for n_vars in list_n_vars:
        #: number of repetitions
        n_qubits = int(np.ceil(np.log2(n_vars)))
        row_dict["n_vars"] = n_vars
        for density in list_densities:
            row_dict["density"] = density
            for id_instance in range(1, max_instances+1):        # id_instance: id of the instance of max_cut
                ## prepare initial data

                row_dict["id_instance"] = id_instance

                G = nx.readwrite.edgelist.read_edgelist("./created_data/random_graphs/rand_graph_N" + str(n_vars) + "-D" + str(density) + "-I"+ str(id_instance) + ".col")    # toy example: 4 nodes, 3 edges
                G.add_nodes_from([str(id_node) for id_node in list(range(n_vars))])

                opt_val = unit_test_max_cut(G, n_qubits, data_log, row_dict)  # this should add multiple rows to data_log (one by iteration)
                print(f"n_vars: {n_vars}, density: {density}, instance: {id_instance}: loc_opt_val: {opt_val}")
                

    # create the pandas DataFrame and save the results
    df = pd.DataFrame(data_log, columns=columns)
    df.to_csv("tests/results/classicalExact_algorithm.csv")       



if __name__ == "__main__":
    #run_unit_test_max_cut(8)
    #state = random_statevector(2**m, 124)                   # np.array
    #unit_test_max_cut(2, shots=1024)
    #unit_test(2)

    #n_qubits_list = [2,3,4]
    #list_n_vars = [4,8,12,16,20,24,32,36]
    list_n_vars = [4,8,12,16,20,24,32,36]
    list_densities = (np.array(list(range(1,10,2)))/10).tolist()  
    max_instances = 10

    run_tests_max_cut(list_n_vars, list_densities,max_instances)