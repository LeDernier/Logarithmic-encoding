#### Set of functions used to solve Max Cut using quantum computers ####

## Importations
import numpy as np
from numpy import pi

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

from qiskit.circuit.library import Diagonal
from qiskit.quantum_info import SparsePauliOp
from qiskit.opflow import CircuitStateFn, PauliSumOp
from qiskit.opflow import CircuitSampler, StateFn, AerPauliExpectation, PauliExpectation

from qiskit import Aer
from qiskit.utils import QuantumInstance

## Functions

def R_map(theta, R_f=None):
    """
        Returns the mapping: 1_{(theta_j mod 2*pi) in [0,pi[}, where theta = (theta_j)_j
    
        ----------
        Parameters
        - theta: an array of real values
        - R_f: resulting vector to be modified. If provided, it can reduce the memory usage.
    """

    R_f_given = True
    if R_f is None:
        R_f = np.zeros(theta.shape)
        R_f_given = False

    for idx, theta_j in np.ndenumerate(theta):
        outOfRange = True
        while outOfRange:
            if theta_j >= 0 and theta_j <= 2*pi:
                outOfRange = False
                break
            
            if theta_j < 0:
                theta_j += 2*pi
            elif theta_j > 2*pi:
                theta_j -= 2*pi

        if theta_j >= 0 and theta_j < pi:
            R_f[idx] = 0
        elif theta_j >= pi and theta_j < 2*pi:
            R_f[idx] = 1

        #R_f[idx] = 2*R_f[idx] - 1
        
            
    if not R_f_given:
        return R_f
        

def Rf_map(theta, m=None, q=0, R_f=None):
    """
        This function is designed to be n-differentiable and for a 
        big value of m, it should represent the function "R_map".

        ----------
        Parameters
        - theta: an array of real values
        - m : >= |V|
        - q : >= 0 (default), <= |V| - 2 
    """
    if m is None:
        m = len(theta)

    R_f = np.zeros(m)

    x_0 = np.arcsin(np.log(-np.log(0.5))/2**m)
    
    for idx, theta_j in np.ndenumerate(theta):
        R_f[idx] = np.exp(-np.exp(2**(m-q) * np.sin(2**q*theta_j + x_0)))
        #R_f[idx] = 2*R_f[idx] - 1                   # mapping {0,1} values to {-1,1} values
        #R_f[idx] = 1 if R_f[idx] >= 0 else -1
    
    return R_f


def Rfs_map(theta, m=None):
    """
        This function is designed to be n-differentiable and for a 
        big value of m, it should represent the function "R_map".
        Here we use a sigmoid function.

        ----------
        Parameters
        - theta: an array of real values
        - m : >= |V|
    """
    if m is None:
        m = len(theta)

    R_f = np.zeros(theta.shape)

    x_0 = pi
    
    for idx, theta_j in np.ndenumerate(theta):
        R_f[idx] = 1/(1 + np.exp(-2**m * np.sin(theta_j - x_0)))    # mapping Reals to a smooth {0,1}
        #R_f[idx] = 2*R_f[idx] - 1                                   # mapping {0,1} values to {-1,1} values
        #R_f[idx] = 1 if R_f[idx] >= 0 else -1
    
    return R_f

def parameterizedCircuit(k, U_diag):
        """
            Returns a circuit that represents the initial state on which the 
            expectation will be calculated.

            ----------
            Parameters
            - k: number of qubits/log-dimension of U
            - U_diag: vector with the entries of U, where U is a diagonal 
                matrix that takes values on {-1,1}
        """

        # initial state
        qr = QuantumRegister(k, name="i")
        qc_init_state = QuantumCircuit(qr)
        qc_init_state.h(range(k))
        qc_init_state.compose(Diagonal(U_diag), inplace=True)

        #qc.decompose(reps=3).draw(output="mpl")
        #qc_init_state.draw(output="mpl")
        #plt.show()
        return qc_init_state

def getExpectedValue(psi, op, backend = None, method = "snapshot", shots=1024, verbose=False):
    """
        Calculate <psi|op|psi>
        ----------
        Parameters
        - psi: quantum state under which the expectation is calculated
        - op: observable (Hermitian matrix)
    """
    
    # define the quantum instance
    q_instance = QuantumInstance(backend, shots=shots)

    # define the state to sample
    measurable_expression = StateFn(op, is_measurement=True).compose(psi)   

    exp_val = None
    if method == "snapshot":
        expectation = AerPauliExpectation().convert(measurable_expression)  # convert to expectation value
        sampler = CircuitSampler(backend).convert(expectation)              # get state sampler        
        exp_val = sampler.eval().real                                       # evaluate  
    elif method == "sampled":
        expectation = PauliExpectation().convert(measurable_expression)     # convert to expectation value
        sampler = CircuitSampler(q_instance).convert(expectation)           # get state sampler
        exp_val = sampler.eval().real                                       # evaluate  
    elif method == "math":
        exp_val = psi.adjoint().compose(op).compose(psi).eval().real        # evaluate  
    
    if verbose:
        print('Expectation[',method,']: ', exp_val)

    return exp_val


def getBindingEnergy(psi, H, U, backend = None, method = "snapshot", shots=512, verbose=False):
    """
        Calculates <+|H*U|psi>.

        Parameters
        - psi: quantum state |+>_n (|++...+>)
        - H: Hamiltonien (Hermitian matrix)
        - U: Control gate (Unitary matrix)
    """
    HU = H @ U
    HU_T = (H @ U).T
    ## decomposition of the symmetric version of H*U in Pauli strings (only works for Hamiltonian with real entries)
    Q_ = (HU + HU_T)/2
    
    isQ_Hermitian = True
    Zero_matrix = Q_ - Q_.T
    for r in Zero_matrix:
        for e in r:
            if e != 0:
                isQ_Hermitian = False

    if not isQ_Hermitian:
        print("H.shape: ", H.shape)
        print("U.shape: ", U.shape)
        print("Q_.shape: ", Q_.shape)
        
        print("U.shape: ", U.shape)
        print("Q_: ", Q_)
    
    pauli_op = SparsePauliOp.from_operator(Q_)
    op = PauliSumOp(pauli_op)
    
    # quantum instance
    q_instance = QuantumInstance(backend, shots=shots)

    # define the state to sample
    measurable_expression = StateFn(op, is_measurement=True).compose(psi)

    exp_val = None
    if method == "snapshot":
        expectation = AerPauliExpectation().convert(measurable_expression)  # convert to expectation value
        sampler = CircuitSampler(backend).convert(expectation)              # get state sampler        
        exp_val = sampler.eval().real                                       # evaluate  
    elif method == "sampled":
        expectation = PauliExpectation().convert(measurable_expression)     # convert to expectation value
        sampler = CircuitSampler(q_instance).convert(expectation)           # get state sampler
        exp_val = sampler.eval().real                                       # evaluate  
    elif method == "math":
        exp_val = psi.adjoint().compose(op).compose(psi).eval().real        # evaluate  
    
    if verbose:
        print('Expectation[',method,']: ', exp_val)

    return exp_val


def getExpectation(H, k, method="sampled", shots=512, verbose=False, R_mapping = "Id"):
    """
        ----------
        Parameters
        - H: Hermitian matrix from the QUBO.
        - k: log-dimension of A
        - method: choose one among three options to calculate the expected value
            'snapshot': the more efficient but bugged for some H. It is like 'sampled' but optimized.
            'sampled': the 2nd more efficient and without bugs (presumably).
            'math': the most inefficient since it is based on matrix multiplication.
        - shots: if method is 'snapshot' or 'sampled', shots is the number of runs/measures of the
            quantum circuit that helps to calculate the expected value. Usually, shots = 512 or 1024.
        - verbose: if True, shows the expected value at each iteration when 
            a solver calls 'parameterizedCircuit'
        - R_mapping: type of mapping from a real value mod 2*pi to {-1,1}.
            'Id' for the identity mapping (default).
            'Rd' for a discrete map: 1_{x in [0,pi[} - 1_{x in [pi,2*pi[}
            'Rc' for the continuous map given in the paper: 2*exp(-exp(2^(m-q)*sin(2^q*x + x_0))) - 1
            'Rcs' for a custom continuous map using a sigmoid: 2*sigmoid(m*(x-pi)) - 1
    """

    # backend
    backend = Aer.get_backend('qasm_simulator') 
    

    ## decomposition of the Hamilonien in Pauli strings
    pauli_op = SparsePauliOp.from_operator(H)
    op = PauliSumOp(pauli_op)
    sum_H = np.sum(H)

    def parameterizedExpectation(theta):
        # optimization parameter
        if R_mapping == "Rd":
            R_f = R_map(theta)
        elif R_mapping == "Rc":
            R_f = Rf_map(theta, 2**k, 0)
        elif R_mapping == "Rcs":
            R_f = Rfs_map(theta, 2**k)
        else:
            # the identity map
            R_f = theta

        U_diag = np.exp(complex(0,np.pi)*R_f)
        
        ## calculations for (1/2^(k-2))*<psi|H|psi>
        # convert to a state
        qc_init_state = parameterizedCircuit(k, U_diag)
        psi = CircuitStateFn(qc_init_state)
        expected_value = getExpectedValue(psi, op, backend=backend, method=method,verbose=verbose)

        if sum_H != 0:  # necessary condition
            ## calculations for (1/2^(k-1))*<+|H*U|+>   (simplest method)
            U_matrix = np.diag(U_diag)
            
            qr = QuantumRegister(k, name="i")
            qc_ones = QuantumCircuit(qr)
            qc_ones.h(range(k))
            psi_ones = CircuitStateFn(qc_ones)
            expected_value_offdiag = getBindingEnergy(psi_ones, H, U_matrix, backend=backend, method=method, shots=shots, verbose=verbose)
            expected_value += 2*expected_value_offdiag
            expected_value += sum_H/4

        return expected_value
    
    return parameterizedExpectation


##### FUNCTIONS TO PERFORM THE TESTS ##########


def get_max_depth(operator_list):
    max_state_prep_detph, max_total_depth = None, None

    try:
        operator_circ = operator_list.to_circuit_op()   # replacing the quantum states by circuits
        circuits = []
        for op in operator_circ.oplist:
            circuits.append(op)


        max_total_depth = max([circ.primitive.decompose(reps=5) for circ in circuits])          # we decompose it to expand the "state preparation" gate
        max_state_prep_detph = max_total_depth - max([circ.primitive for circ in circuits])
    except:
        pass

    return max_state_prep_detph, max_total_depth

    