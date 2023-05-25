import numpy as np
import networkx as nx
from os import path

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


def readGraph(file):
    """ 
        Read a graph instance from a file '.col'.

        ----------
        Parameters
        - file: path of the file

        -------
        Return
        - number of nodes
        - number of edges
        - adjacency matrix
    """

    if not path.isfile(file):
        print("The file to construct the graph could not be found.")
        return

    myFile=open(file, "r")
    data=myFile.readlines()
    line=data[0].split(" ")
    n=int(line[2]) # number of nodes
    m=int(line[3]) # number of edges
    adj=np.zeros((n,n))       # adjacency matrix

    for i in range(1,1+m):
        line=data[i].split(" ")
        u=int(line[1])
        v=int(line[2])
        adj[u-1,v-1]=1
        adj[v-1,u-1]=1

    return n,m,adj
