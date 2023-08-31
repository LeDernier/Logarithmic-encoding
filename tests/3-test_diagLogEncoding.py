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
from qiskit.circuit import Parameter, ParameterVector           # TODO: change to parameterized circuits



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



# Functions for diagonalizing matrices


##### CUSTOM CODE FOR MAX CUT #######

def evaluate(N, beta, counts, shots):
    """
        It calculates the expectation value after having mapped the Paulis to the basis {I,Z} and
        having measured the corresponding quantum circuits.

        Parameters
        - beta: coefficients associated to the set of pairwise commuting families that have been mapped to the{I,Z} basis.
        - counts: results from the quantum circuits.
        - shots: number of shots of the quantum circuits.
    """
    
    
    if isinstance(counts, list):
        new_counts = np.zeros((len(counts),N))
        for fam in range(len(counts)):
            for key,value in counts[fam].items():
                new_counts[fam, int(key,2)] = value
    else:
        new_counts = np.zeros((1,N))
        for key,value in counts.items():
            new_counts[0, int(key,2)] = value

    exp_val = np.sum(np.multiply(beta, new_counts))/shots

    return exp_val




def unit_test_max_cut_ONLINE(state, m, N, shots, Hmat,  post_circuits, beta,
                             data_log, row_dict, backend = Aer.get_backend('aer_simulator')):
    """ 
        Performs calculations during the optimization procedure (ONLINE). It usually depends on the values of the 
        vector of variables.
        
        Parameters:
            - state: np.array() = vector of variables (x_1,...,x_{N}). The vector is not unitary, it takes 
                values in \{-1,1\}^{2^m}.
            - m: int = number of qubits.
            - N: int = 2**m.
            - shots: number of shots (executions of the quantum circuits).
            - Hmat: np.array() = QUBO matrix.
            - post_circuits: list = circuits that will be composed with the quantum state |x_1,...,x_{N}>.
            - beta: np.array() = offline coefficients to calculate the expectation value. 
            - backend: quantum backend used during the transpilation phase. 

        Returns:
            - expectation: value of the objective function for a particular vector of variables "state".
    """
    """ ### projecting the vector of variables to the feasible space
    state = np.sign(state)
    for i in range(len(state)):
        if state[i] == 0:
            state[i] == 1 """
    
    t_start_Rf_map = perf_counter()
    #R_f = Rfs_map(state, 1000)              # sigmoid [first function tested]
    R_f = Rf_map(state)                    # continuous (from the paper) [second function tested]
    #R_f = R_map(state)                      # discrete
    state = np.exp(complex(0,np.pi)*R_f)
    time_Rf_map = perf_counter() - t_start_Rf_map

    ### pure classical evaluation ("direct evaluation")
    t_start_de = perf_counter()     
    direct_eval = (state.conjugate().T @ Hmat @ state).real
    time_direct_eval = perf_counter() - t_start_de

    
    ### (hybrid classical-)quantum evaluation 
    # preparation of the quantum state |++...+>
    t_start_state = perf_counter()
    qc_init = QuantumCircuit(m)
    for i in range(m):
        qc_init.h(i)


    # preparation of the quantum state |x_1,...,x_{2^m}>
    state_circuit = Diagonal(state).decompose(reps=3)            # TODO: check if it is really necessary to have the "StateVector" (we are only using the data)
    time_init_state = perf_counter() - t_start_state

    t_start_comp = perf_counter()    # is: initial state
    state_circuit = transpile(qc_init.compose(state_circuit.decompose(reps=3)), backend=backend)

    circuits = [state_circuit.compose(p_circ).measure_all(inplace=False) for p_circ in post_circuits]      # the "measure_all()" is necessary to use the sampler
    #transpiled_circuits = transpile(circuits, backend=backend)
    time_comp_state = perf_counter() - t_start_comp          
    depth_init_state = state_circuit.depth()        # depth of the circuit preparing the state encoding the vector of variables

    
    ## 
    
    for i in range(len(circuits)):
        circuits[i].name = "Circuit_m"+str(m)+"_f"+str(i)    # m: number of qubits, f: family
    
    t_start_run = perf_counter()
    job = backend.run(circuits, shots=shots)
    counts_dict = job.result().get_counts()
    time_qc_run = perf_counter() - t_start_run
    
    """ 
    # TODO: remove this 6 lines since the results are already ordered 
    # initialize list to store ordered results
    counts_ordered = [None] * len(transpiled_circuits)

    for i in range(len(transpiled_circuits)):
        name = result_dict[i]["header"]["name"]
        n = int(name.split('_f')[1])  # index of circuit in input list
        counts_ordered[n] = counts_dict[i]  # add to result list at same index   """
    
    # calculate the expectation value from the measures and offline coefficients (beta)
    
    t_start_classic = perf_counter()
    new_eval = evaluate(N, beta, counts_dict, shots).real*N
    time_classic_evaluation = perf_counter() - t_start_classic
   
    #print(f"{direct_eval = }")
    #print(f"{new_eval = }")

    ## record  
    row_dict["id_iter"] += 1
    row_dict["real_value"] = direct_eval
    row_dict["estimated_value"] = new_eval
    row_dict["depth_init_state"] = depth_init_state

    row_dict["timeON_Rf_map"] = time_Rf_map
    row_dict["timeON_qc_construction"] = time_init_state
    row_dict["timeON_qc_compilation"] = time_comp_state
    row_dict["timeON_qc_run"] = time_qc_run
    row_dict["timeON_classic_eval"] = time_classic_evaluation

    row_dict["time_direct_eval"] = time_direct_eval

    """ # to return: "real_value", "error", "max_circuit_depth", "quantum_time", "total_time"
    values = [direct_eval.real, new_eval]
    depths = [depth_init_state, max_depth_post_qc, ave_depth_post_qc]
    times = [time_prep_qs, time_transpilation, time_execution_qc, time_classic_evaluation]  # qs: quantum state, qc: quantum circuit """

    data_log.append(list(row_dict.values()))

    return new_eval

def run_unit_test_max_cut_OFFLINE(Hmat, m, row_dict, backend = Aer.get_backend('aer_simulator')):
    """
        Performs calculations before the optimization procedure takes place (OFFLINE). That is: grouping the set of pairwise
        commuting families of Paulis, mapping them to the basis {I,Z} by increasing the depth of the quantum circuit, and 
        also calculating some coefficients "beta" that will be use to calculate the expectation value. 
        
        Parameters:
            - Hmat: QUBO matrix associated to the instance of Max-Cut.
            - m: number of qubits
            - backend: quantum backend used during the transpilation phase.

        Returns:
            - post_circuits: circuits to be measured so the quasi-probabilities can be estimated via sampling.
            - beta: coefficients that can be calculated "offline".
    """
    
    densePauliExp = DensePauliExpectation(group_paulis=True)

    """
    H = get_Op(Hmat, 'naive')   # decomposition in the Pauli basis [SparsePauliOp()]

    t_start_qc_basis = perf_counter()
    expectation = densePauliExp.convert(H)              # list of pairs (mapped operator, transformation matrix between two basis as a circuit)
    mapped_operators = [comp_op[0].simplify() for comp_op in expectation]   # Pauli operators mapped to the {I,Z} basis
    time_qc_basis = perf_counter() - t_start_qc_basis
    """

    t_start_diag = perf_counter()
    D_diag, P = np.linalg.eigh( Hmat )
    D = np.diag(D_diag)
    P = np.matrix(P)
    time_diagonalization = perf_counter() - t_start_diag
    
    ### decomposition of D, the diagonalization of Hmat
    

    H_D = get_Op(D, 'naive').simplify()
    t_start_qc_basis = perf_counter()
    expectation = densePauliExp.convert(H_D)       # list of pairs (mapped operator, transformation matrix between two basis as a circuit)
    mapped_operators = [comp_op[0] for comp_op in expectation]   # Pauli operators mapped to the {I,Z} basis. It should be a singleton since H_D is diagonal.
    time_qc_basis = perf_counter() - t_start_qc_basis

    ## OFFLINE coefficients
    t_start_coeffs = perf_counter()
    beta = [] # matrix (i,l) coefficients associated to the family J_i, eigenstate l : sum_{k in J_i} diag(Z_{sigma(k)})_l * C_{m,k}
    for oper_fam in mapped_operators:
        beta_fam = np.diag(oper_fam.to_matrix())   # coefficient associated to one family
        beta.append(beta_fam)
    beta = np.array(beta)
    time_off_coeffs = perf_counter() - t_start_coeffs
    
    
    ### implement the unitary matrix P
    
    
    post_circuits = []      # circuits to compose with "state" for each commuting family mapped (transformation between two basis) 
    for comp_op in expectation:
        if isinstance(comp_op[1].primitive, QuantumCircuit):
            post_circuits.append(comp_op[1].primitive)
        else:
            post_circuits.append(QuantumCircuit(row_dict["n_qubits"]))
    
    P_oper = UnitaryGate(P)
    for circ in post_circuits:
        circ.append(P_oper, list(range(m)))
    

    post_circuits = [circ.decompose(reps=3) for circ in post_circuits]
    t_start_comp = perf_counter() 
    post_circuits = transpile(post_circuits, backend=backend)
    time_comp_state = perf_counter() - t_start_comp
    

    ## depths
    depth_post_circuits = [circ.depth() for circ in post_circuits]
    row_dict["max_depth_post_qc"] = max(depth_post_circuits)
    row_dict["min_depth_post_qc"] = min(depth_post_circuits)
    row_dict["ave_depth_post_qc"] = np.mean(depth_post_circuits)      # average of the depths
    row_dict["std_depth_post_qc"] = np.std(depth_post_circuits)                                 # standard deviation of the depths

    row_dict["timeOFF_qc_basis"] = time_qc_basis
    row_dict["timeOFF_coeffs"] = time_off_coeffs
    row_dict["timeOFF_diagonalization"] = time_diagonalization
    row_dict["timeOFF_compilation"] = time_comp_state

    return post_circuits, beta


def unit_test_max_cut(Hmat, m, shots, data_log, row_dict, backend = Aer.get_backend('aer_simulator')):
    """ 
        Resolution of the instance of max-cut approximatively. This function modifies a dictionary and 
        a list that will be used to record the results.

        Parameters:
            - m: number of qubits.

        Returns:
            - Optimal value.
    """

    N = 2**m
    post_circuits, beta = run_unit_test_max_cut_OFFLINE(Hmat, m, row_dict)
    row_dict["beta_max"] = np.max(np.abs(beta))
    
    vars = np.zeros(N)
    expectation = partial(unit_test_max_cut_ONLINE, m=m, N=N, shots=shots, Hmat=Hmat, 
                          post_circuits=post_circuits, beta=beta, backend=backend, 
                          data_log=data_log, row_dict=row_dict)

    res = minimize(expectation, 
                      vars, 
                      method='COBYLA')

    return res.fun, res.x



def run_tests_max_cut(list_n_vars: list, list_densities: list, max_instances = 10, n_shots = 512, 
                      backend = Aer.get_backend('aer_simulator')):
    """
        list_n_vars: list with the number of nodes to test.
        list_densities: list with the number of densities to test.
        max_instances: maximum number of repetitions of the evaluation of a quantum circuit
        n_shots: maximum number of shots of each quantum circuit
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
    cols_data = ["n_qubits", "n_vars", "density", "n_shots", "id_instance", "beta_max", "id_iter"]
    cols_values = ["real_value", "estimated_value"]
    cols_depths = ["depth_init_state","min_depth_post_qc", "ave_depth_post_qc", "max_depth_post_qc", "std_depth_post_qc"]
    cols_times = ["timeOFF_qc_basis","timeOFF_coeffs", "timeOFF_diagonalization", "timeOFF_compilation", "timeON_Rf_map", 
                  "timeON_qc_construction", "timeON_qc_compilation", "timeON_qc_run", "timeON_classic_eval", "time_direct_eval"]
    columns = cols_data + cols_values + cols_depths + cols_times

    data_log = []      # mutable object to store the 
    row_dict = {columns[i]: None for i in range(len(columns))}
    
    row_dict["n_shots"] = n_shots

    # n_qubits: number of qubits
    for n_vars in list_n_vars:
        #: number of repetitions
        n_qubits = int(np.ceil(np.log2(n_vars)))
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
                """ n_edges = G.number_of_edges()
                if n_edges > 0: """
                #Hmat = (-2**(n_qubits-2))*get_laplacian_matrix(G)             # Observable
                Hmat = (-2**(-2))*get_laplacian_matrix(G)             # Observable
                Hmat = correct_dims(Hmat) 

                opt_val = unit_test_max_cut(Hmat, n_qubits, n_shots, data_log, row_dict)  # this should add multiple rows to data_log (one by iteration)
                print(f"n_vars: {n_vars}, density: {density}, instance: {id_instance}: loc_opt_val_estimated: {data_log[len(data_log)-1][7]}, loc_opt_val_real: {data_log[len(data_log)-1][6]}")
                    


    # create the pandas DataFrame and save the results
    df = pd.DataFrame(data_log, columns=columns)
    df.to_csv("tests/results/maxCut2_test_DiagLogEncoding_algorithm.csv")            


if __name__ == "__main__":
    #run_unit_test_max_cut(8)1
    #state = random_statevector(2**m, 124)                   # np.array
    #unit_test_max_cut(2, shots=1024)
    #unit_test(2) 

    #n_qubits_list = [2,3,4]
    """ list_n_vars = [4,8,12,16,20,24,32]
    max_instances = 10 """
    #list_n_vars = [36,48]
    list_n_vars = [12]
    list_densities = (np.array(list(range(1,10,2)))/10).tolist()  
    max_instances = 5
    n_shots = 1024

    run_tests_max_cut(list_n_vars, list_densities, max_instances, n_shots)
    """ matrix = np.array([[1,1,1,1],
                       [1,-1,1,-1],
                       [1,1,-1,-1],
                       [1,-1,-1,1]])

    evals, evecs = np.linalg.eigh( matrix )
    D2 = np.diag(evals)
    P2 = np.matrix(evecs)

    print(f"evals: {evals}, evecs: {evecs}")
    print(f"D2:\n {D2}\nP2:\n {P2}\n(P2)(P2)^T:\n{P2 @ P2.H}") """