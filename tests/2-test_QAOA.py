import sys
import os
import numpy as np
import scipy.linalg
import pandas as pd
from numpy.random import default_rng
from typing import List, Tuple, Union, cast
from time import time, perf_counter
import networkx as nx

from qiskit import Aer, transpile, QuantumCircuit
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
from qiskit.extensions import UnitaryGate
from qiskit.circuit import Parameter, ParameterVector

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
from graph_functions import get_laplacian_matrix, get_adjacency_matrix
from expectation_functions import get_max_depth, R_map, Rf_map, Rfs_map



# Functions for diagonalizing matrices


##### CUSTOM CODE FOR MAX CUT #######


def maxcut_obj(solution, graph):
    """Given a bit string as a solution, this function returns
    the number of edges shared between the two partitions
    of the graph (multiplied by -1 because we are minimizing).
    Args:
        solution: (str) solution bit string
        graph: networkx graph
    Returns:
        obj: (float) Objective
    """
    
    obj = 0
    for i, j in graph.edges():
        if solution[int(i)] != solution[int(j)]:
            #obj -= graph[i][j]["weight"]
            obj -= 1
    return obj




def compute_expectation(counts, graph):
    """Computes expectation value based on measurement results
    Args:
        counts: (dict) key as bit string, val as count
        graph: networkx graph
    Returns:
        avg: float
             expectation value
    """

    avg = 0
    sum_count = 0
    for bit_string, count in counts.items():
        obj = maxcut_obj(bit_string, graph)
        avg += obj * count
        sum_count += count
    return avg/sum_count


# We will also bring the different circuit components that
# build the qaoa circuit under a single function
def create_qaoa_circ(N, graph, theta):
    """Creates a parametrized qaoa circuit
    Args:
        graph: networkx graph
        theta: (list) unitary parameters
    Returns:
        (QuantumCircuit) qiskit circuit
    """
    #nqubits = len(graph.nodes())
    nqubits = N
    edges = [(int(u),int(v)) for u,v in graph.edges()]
    n_layers = len(theta)//2  # number of alternating unitaries
    beta = theta[:n_layers]
    gamma = theta[n_layers:]

    qc = QuantumCircuit(nqubits)

    # initial_state
    qc.h(range(nqubits))

    for layer_index in range(n_layers):
        # problem unitary
        for pair in list(edges):
            qc.rzz(2 * gamma[layer_index], pair[0], pair[1])
        # mixer unitary
        for qubit in range(nqubits):
            qc.rx(2 * beta[layer_index], qubit)

    qc.measure_all()
    return qc


# Finally we write a function that executes the circuit
# on the chosen backend
def get_expectation(graph, dim, data_log, row_dict, shots=512, n_layers=1, backend = Aer.get_backend('aer_simulator')):
    """Runs parametrized circuit
    Args:
        graph: networkx graph
    """

    ## OFFLINE computations

    theta_param = ParameterVector("theta", 2*n_layers)
    t_start_state = perf_counter()
    qc = create_qaoa_circ(dim, graph, theta_param)
    time_quantum_state = perf_counter() - t_start_state

    t_start_comp = perf_counter() 
    compiled_qc = transpile(qc, backend=backend)
    time_comp_state = perf_counter() - t_start_comp
    
    depth_qc = compiled_qc.depth()
    
    row_dict["depth_qc"] = depth_qc                                 # depth of the parameterized quantum circuit
    row_dict["timeOFF_qc_compilation"] = time_comp_state
    row_dict["timeOFF_qc_construction"] = time_quantum_state

    def execute_circ(theta):
        ## ONLINE computations

        #qc = create_qaoa_circ(graph, theta)
        t_start_run = perf_counter()
        try:
            binded_qc = compiled_qc.bind_parameters({theta_param: theta})
        except:
            # continue if the parameter is not in the circuit
            pass
        job = backend.run(binded_qc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        time_qc_run = perf_counter() - t_start_run

        
        ## estimate the objective value
        t_start_classic = perf_counter()
        new_eval = compute_expectation(counts, graph)                # estimated expectation value
        time_classic_evaluation = perf_counter() - t_start_classic

        row_dict["id_iter"] += 1
        row_dict["estimated_value"] = new_eval
        idx_basis_state = np.argmax(list(counts.values()))       # position of the basis state with higher estimated probability
        row_dict["bin_basis_state"] = list(counts.keys())[idx_basis_state]
        row_dict["prob_basis_state"] = counts[row_dict["bin_basis_state"]]/shots
    
        row_dict["timeON_qc_run"] = time_qc_run
        row_dict["timeON_classic_eval"] = time_classic_evaluation

        data_log.append(list(row_dict.values()))

        #print("iter: ", row_dict["id_iter"], ", value: ", new_eval, ",\n\ttime_classic_eva: ", time_classic_evaluation)

        return new_eval

    return execute_circ


####################################################################


def unit_test_max_cut(graph, m, shots, n_layers, data_log, row_dict, backend = Aer.get_backend('aer_simulator')):
    """
        Parameters:
            - graph [nx.Graph()] = graph
            - m [int] = number of qubits.
            - shots: number of shots
            - n_layers: number of layers.
    """

    N = m
    n_edges = graph.number_of_edges()

    if n_edges > 0:

        vars = np.zeros(2*n_layers).tolist()

        expectation = get_expectation(graph, N, data_log, row_dict, shots, n_layers, backend)
        res = minimize(expectation,
                    vars,
                    method='COBYLA')
        

        ## calculating exactly the value of the objective function for the local optima (assuming it is of the form e_j)
        x_opt_string = data_log[len(data_log) - 1][7]             # position of basis state with highest probability: 6
        x_opt_string_rev = list(reversed(x_opt_string))
        x_opt = np.zeros(N)
        x_opt[:len(x_opt_string_rev)] = x_opt_string_rev
        y_opt = 1 - 2*x_opt

        Hmat = (-2**(-2))*get_laplacian_matrix(graph)             # Observable
        #y_opt_red = y_opt.take(graph.nodes(), axis=0)
        #real_value = y_opt_red.T @ Hmat  @ y_opt_red
        real_value = y_opt.T @ Hmat  @ y_opt
        data_log[len(data_log)-1][10] = real_value                 # position of real_value in data_log: 9
        
        return res.fun
    
    return 0.0


def run_tests_max_cut(list_n_vars: list, list_densities: list, max_instances = 10, n_shots = 512, 
                      n_layers = 1, backend = Aer.get_backend('aer_simulator')):
    """
        list_n_vars: list with the number of nodes to test.
        list_densities: list with the number of densities to test.
        max_instances: maximum number of repetitions of the evaluation of a quantum circuit
        n_shots: maximum number of shots of each quantum circuit
        n_layers: number of consecutive layers "problem Hamiltonian * mixing Hamiltonian".
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
    cols_data = ["n_layers", "n_qubits", "n_vars", "density", "n_shots", "id_instance", "id_iter"]
    cols_values = ["bin_basis_state", "prob_basis_state", "estimated_value", "real_value"]
    cols_depths = ["depth_qc"]
    cols_times = ["timeOFF_qc_construction", "timeOFF_qc_compilation", "timeON_qc_run", "timeON_classic_eval"]  # qs: quantum state
    columns = cols_data + cols_values + cols_depths + cols_times

    data_log = []      # mutable object to store the 
    row_dict = {columns[i]: None for i in range(len(columns))}
    row_dict["n_layers"] = n_layers
    row_dict["n_shots"] = n_shots

    # n_qubits: number of qubits
    for n_vars in list_n_vars:
        #: number of repetitions
        n_qubits = n_vars
        row_dict["n_qubits"] = n_qubits
        row_dict["n_vars"] = n_vars
        for density in list_densities:
            row_dict["density"] = density
            for id_instance in range(1, max_instances+1):        # id_instance: id of the instance of max_cut
                ## prepare initial data

                row_dict["id_instance"] = id_instance
                row_dict["id_iter"] = 0

                #G = nx.readwrite.edgelist.read_weighted_edgelist("./created_data/random_graphs/rand_graph_" + str(N) + "."+ str(id_instance) + ".col")   
                G = nx.readwrite.edgelist.read_edgelist("./created_data/random_graphs/rand_graph_N" + str(n_vars) + "-D" + str(density) + "-I"+ str(id_instance) + ".col")    # toy example: 4 nodes, 3 edges
                G.add_nodes_from([str(id_node) for id_node in list(range(n_vars))])
                #Hmat = (-2**(n_qubits-2))*get_laplacian_matrix(G)             # Observable
                #Hmat = (-2**(-2))*get_laplacian_matrix(G)             # Observable

                opt_val = unit_test_max_cut(G, n_qubits, n_shots, n_layers, data_log, row_dict)  # this should add multiple rows to data_log (one by iteration)
                print(f"n_vars: {n_vars}, density: {density}, instance: {id_instance}: loc_opt_val_estimated: {opt_val}")

    # create the pandas DataFrame and save the results
    df = pd.DataFrame(data_log, columns=columns)
    df.to_csv("tests/results/maxCut2_32_QAOA_algorithm_p"+str(n_layers)+".csv")            




if __name__ == "__main__":
    #run_unit_test_max_cut(8)
    #state = random_statevector(2**m, 124)                   # np.array
    #unit_test_max_cut(2, shots=1024)
    #unit_test(2)

    n_layers = 1
    #n_qubits_list = [2,3,4]
    #list_n_vars = [4,8,12,16,20,24]
    list_n_vars = [32]
    list_densities = (np.array(list(range(1,10,2)))/10).tolist()
    max_instances = 10
    

    n_shots = 1024

    run_tests_max_cut(list_n_vars, list_densities, max_instances, n_shots, n_layers)
    """ matrix = np.array([[1,1,1,1],
                       [1,-1,1,-1],
                       [1,1,-1,-1],
                       [1,-1,-1,1]])

    evals, evecs = np.linalg.eigh( matrix )
    D2 = np.diag(evals)
    P2 = np.matrix(evecs)

    print(f"evals: {evals}, evecs: {evecs}")
    print(f"D2:\n {D2}\nP2:\n {P2}\n(P2)(P2)^T:\n{P2 @ P2.H}") """