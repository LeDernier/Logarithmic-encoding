import sys
import os
import numpy as np
import pandas as pd
from numpy.random import default_rng
from typing import List, Tuple, Union, cast
from time import time, perf_counter
import networkx as nx

from qiskit import Aer, transpile, QuantumCircuit
#from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Sampler, Session, Options   # TODO: change the version of qiskit before using qiskit_ibm_

from qiskit.opflow import StateFn, DictStateFn, PauliExpectation, CircuitSampler
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.list_ops.summed_op import SummedOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.state_fns.operator_state_fn import OperatorStateFn
from qiskit.opflow.primitive_ops.pauli_op import PauliOp

#from qiskit.quantum_info import random_statevector
from qiskit.utils import QuantumInstance
from qiskit.quantum_info import Statevector, SparsePauliOp, Pauli
from qiskit.circuit.library import Diagonal



## Non-Qiskit
from dense_ev.rmatrix import random_H, get_Op
from dense_ev.dense_pauli_expectation import DensePauliExpectation

from scipy.optimize import minimize
from collections import defaultdict
from psfam.pauli_organizer import PauliOrganizer
from dense_ev.decompose_pauli import to_pauli_vec
from functools import partial


## Imports customs
sys.path.append('./code')
from graph_functions import get_laplacian_matrix, get_adjacency_matrix, correct_dims
from expectation_functions import get_max_depth, R_map, Rf_map, Rfs_map


##### CUSTOM CODE FOR MAX CUT #######


def unit_test_max_cut_ONLINE(state, Hmat, data_log, row_dict):
    """ 
        Performs calculations during the optimization procedure (ONLINE). It usually depends on the values of the 
        vector of variables.
        
        Parameters:
            - state: np.array() = vector of variables (x_1,...,x_{N}). The vector is not unitary, it takes 
                values in \{-1,1\}^{2^m}.
            - Hmat: np.array() = QUBO matrix.

        Returns:
            - expectation: value of the objective function for a particular vector of variables "state".
    """
    """ ### projecting the vector of variables to the feasible space
    state = np.sign(state)
    for i in range(len(state)):
        if state[i] == 0:
            state[i] == 1 """
    
    t_start_Rf_map = perf_counter()  
    R_f = Rf_map(state)                    # continuous (from the paper)
    #R_f = R_map(state)                      # discrete
    state = np.exp(complex(0,np.pi)*R_f)
    time_Rf_map = perf_counter()   - t_start_Rf_map

    ### pure classical evaluation ("direct evaluation")
    t_start_de = perf_counter()       
    direct_eval = (state.conjugate() @ Hmat @ state).real
    time_direct_eval = perf_counter()   - t_start_de

    
    ## record  
    row_dict["id_iter"] += 1
    row_dict["real_value"] = direct_eval
    row_dict["estimated_value"] = direct_eval

    row_dict["timeON_Rf_map"] = time_Rf_map
    row_dict["time_direct_eval"] = time_direct_eval

    data_log.append(list(row_dict.values()))

    return direct_eval


def unit_test_max_cut(Hmat, m, data_log, row_dict):
    """
        Resolution of the instance of max-cut approximatively. This function modifies a dictionary and 
        a list that will be used to record the results.

        Parameters:
            - m: number of qubits.

        Returns:
            - Optimal value.
    """

    N = m
    #post_circuits, beta = run_unit_test_max_cut_OFFLINE(Hmat, row_dict)
    vars = np.zeros(N)
    expectation = partial(unit_test_max_cut_ONLINE, Hmat=Hmat, 
                          data_log=data_log, row_dict=row_dict)

    res = minimize(expectation, 
                      vars, 
                      method='COBYLA')
    
    return res.fun




def run_tests_max_cut(list_n_vars: list, list_densities: list, max_instances = 10):
    """
        list_n_vars: list with the number of nodes to test.
        list_densities: list with the number of densities to test.
        max_instances: maximum number of repetitions of the evaluation of a quantum circuit.
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
    cols_data = ["n_qubits", "density", "n_vars", "id_instance", "id_iter"]
    cols_values = ["real_value", "estimated_value"]
    cols_depths = []
    cols_times = ["timeON_Rf_map", "time_direct_eval"]
    columns = cols_data + cols_values + cols_depths + cols_times

    data_log = []      # mutable object to store the 
    row_dict = {columns[i]: None for i in range(len(columns))}

    # n_qubits: number of qubits
    for n_vars in list_n_vars:
        #: number of repetitions
        #n_qubits = int(np.ceil(np.log2(n_vars)))
        n_qubits = n_vars
        row_dict["n_qubits"] = n_qubits
        row_dict["n_vars"] = n_vars
        for density in list_densities:
            row_dict["density"] = density
            for id_instance in range(1, max_instances+1):        # id_instance: id of the instance of max_cut
                ## prepare initial data

                row_dict["id_instance"] = id_instance
                row_dict["id_iter"] = 0

                G = nx.readwrite.edgelist.read_edgelist("./created_data/random_graphs/rand_graph_N" + str(n_vars) + "-D" + str(density) + "-I"+ str(id_instance) + ".col")    # toy example: 4 nodes, 3 edges
                G.add_nodes_from([str(id_node) for id_node in list(range(n_vars))])
                #Hmat = (-2**(n_qubits-2))*get_laplacian_matrix(G)             # Observable
                Hmat = (-2**(-2))*get_laplacian_matrix(G)             # Observable
                #Hmat = correct_dims(Hmat)                            # TODO: check change of dimensions was done correctly: 2**m --> m

                opt_val = unit_test_max_cut(Hmat, n_qubits, data_log, row_dict)  # this should add multiple rows to data_log (one by iteration)
                print(f"n_vars: {n_vars}, density: {density}, instance: {id_instance}: loc_opt_val: {opt_val}")
                    


    # create the pandas DataFrame and save the results
    df = pd.DataFrame(data_log, columns=columns)
    df.to_csv("tests/results/maxCut2_v2_104-116-128-192-256-_classicalApproximation_algorithm.csv")            



if __name__ == "__main__":
    #run_unit_test_max_cut(8)
    #state = random_statevector(2**m, 124)                   # np.array
    #unit_test_max_cut(2, shots=1024)
    #unit_test(2)

    #n_qubits_list = [2,3,4]
    #list_n_vars = [4,8,12,16,20,24,32,36,48,64]
    #list_n_vars = [56,72,84,96]
    list_n_vars = [104,116,128,192,256]
    list_densities = (np.array(list(range(1,10,2)))/10).tolist()  
    max_instances = 10

    run_tests_max_cut(list_n_vars, list_densities, max_instances)