"""
Goemans & Williamson classical algorithm for MaxCut
"""

from typing import Tuple
from time import time

import cvxpy as cvx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# custom modules
from graph_functions import get_laplacian_matrix
nx.Graph()


def goemans_williamson(graph):
    """
    The Goemans-Williamson algorithm for solving the maxcut problem.

    Ref:
        Goemans, M.X. and Williamson, D.P., 1995. Improved approximation
        algorithms for maximum cut and satisfiability problems using
        semidefinite programming. Journal of the ACM (JACM), 42(6), 1115-1145
    Returns:
        np.ndarray: Graph partition (+/-1 for each node)
        float:      The GW score for this cut
        float:      The GW bound from the SDP relaxation
        float:      Execution time in seconds
    """
    t_start = time()

    laplacian = np.array(0.25 * nx.laplacian_matrix(graph).todense())

    ## setup and solve the GW semidefinite programming problem
    psd_mat = cvx.Variable(laplacian.shape, PSD=True)   # positive_semidefinite_matrix
    obj = cvx.Maximize(cvx.trace(laplacian @ psd_mat))
    constraints = [cvx.diag(psd_mat) == 1]  # unit norm
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver=cvx.CVXOPT)

    evals, evects = np.linalg.eigh(psd_mat.value)
    sdp_vectors = evects.T[evals > float(1.0E-6)].T

    # bound from the SDP relaxation
    bound = np.trace(laplacian @ psd_mat.value)     # value of the SDP relaxation

    ## Goemans and Williamson approximation technique
    random_vector = np.random.randn(sdp_vectors.shape[1])   # random normal
    random_vector /= np.linalg.norm(random_vector)
    signs = np.sign([vec @ random_vector for vec in sdp_vectors])
    score = signs @ laplacian @ signs.T             # value of the rounded solution

    elapsed_time = np.round(time() - t_start,5)

    return signs, score, bound, elapsed_time


if __name__ == '__main__':
    # quick test of this algorithm
    G = nx.readwrite.edgelist.read_weighted_edgelist("../created_data/toy_graph_4.col")    # toy example: 4 nodes, 3 edges
    num_nodes = G.number_of_nodes()
    nodes_labels = [int(node) for node in G.nodes]
    laplacian = np.array(0.25 * get_laplacian_matrix(G))

    """ pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos)
    plt.show() """
    print("nodes: ", )
    #print("laplacian:\n", 4*laplacian)

    result = goemans_williamson(G)
    bound = result[2]
    score = result[1]
    solution = result[0]
    solution_time = result[3]
    solution_ord = [solution[x] for _,x in sorted(zip(nodes_labels,list(range(num_nodes))))]
    print("bound: ", bound)
    print("score: ", score)
    print("solution: ", solution_ord)
    print("solution time: ", solution_time)

    #assert np.isclose(bound, 36.25438489966327)
    print(goemans_williamson(G))

    scores = [goemans_williamson(G)[1] for n in range(100)]
    #assert max(scores) >= 34

    print(min(scores), max(scores))