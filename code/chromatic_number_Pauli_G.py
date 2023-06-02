import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import grinpy as gp
from pulp import *

def create_Pauli_graph(n:int):
    """
        Parameters:
            - n: string size or number of qubits.
    """
    assert n >= 1, "n should be greater than zero"

    OnePauliNodes = ["I","X","Y","Z"]

    currentPauliNodes = ["I","X","Y","Z"]
    NextPauliNodes = []
    for d in range(1,n):
        for P in currentPauliNodes:
            for Q in OnePauliNodes:
                NextPauliNodes.append(P+Q)
        currentPauliNodes = NextPauliNodes
        NextPauliNodes = []
    
    currentPauliEdges = []
    num_nodes = len(currentPauliNodes)
    for i in range(num_nodes):
        P = currentPauliNodes[i]
        for j in range(i+1,num_nodes):
            Q = currentPauliNodes[j]
            num_com_terms = 0   # number of anti-commuting terms
            for idx in range(n):
                if P[idx] != "I" and Q[idx] != "I" and P[idx] != Q[idx]:
                    num_com_terms += 1
            
            if num_com_terms % 2 == 1:
                currentPauliEdges.append((P,Q))
            
    G = nx.Graph()
    G.add_nodes_from(currentPauliNodes)
    G.add_edges_from(currentPauliEdges)

    return G

def get_chromatic_number(G:nx.Graph):
    return gp.chromatic_number(G)

def get_minimum_number_of_colors(G, n:int):
    """
        Parameters
         - G: the Pauli graph
         - n: the string size/number of qubits
    """
    colors = list(range(int(0.5*(3**n + 3))))    # list of possible colors (using the upper bound)
    #colors = list(range(4**n))
    nodes = G.nodes
    edges = G.edges

    # model
    model = LpProblem("vertex_coloring", sense=const.LpMinimize)
    possible_coloring = [(u,c) for u in nodes for c in colors]

    x = LpVariable.dicts("node_color", possible_coloring, 0, 1, LpInteger)
    y = LpVariable.dicts("color", colors, 0, 1)
    model += (
        lpSum([y[c] for c in colors]),
        "number_of_colors"
    )

    for u in nodes:
        model += (lpSum([x[u,c] for c in colors]) == 1, 
                  "node_colored_"+str(u))
        
        for c in colors:
            model += (x[u,c] <= y[c], "color_used_"+str((u,c)))
        
    for u,v in edges:
        for c in colors:
            model += (x[u,c] + x[v,c] <= 1,
                      "different_color_"+str((u,v,c)))
    
        
    #print("MODEL:\n", model)
    
    solver = getSolver("CPLEX_CMD")
    model.solve(solver)
    obj_val = value(model.objective)
    print("number_of_colors: ", obj_val)
    color_group = []
    for v in nodes:
        for c in colors:
            if abs(value(x[v,c]) - 1) < 0.001:
                color_group.append((c,v))

    return obj_val, color_group
        

if __name__  == '__main__':
    n = 2
    G = create_Pauli_graph(n)
    #chromatic_num = get_chromatic_number(G)
    chromatic_num, coloring = get_minimum_number_of_colors(G,n)
    print(f"chromatic number for {n} qubits: ", chromatic_num)
    print("coloring: ", coloring)

    coloring_s = sorted(coloring)
    current_color = -1
    lines = ""
    for color,node in coloring_s:
        if color != current_color:
            lines += "\n" + str(color) + ": "
            current_color = color
        lines += node+", "

    print(lines)

    cmap = plt.get_cmap('hsv')
    colors = cmap(np.linspace(0.0, 1.0, int(chromatic_num)+2))
    color_map = [color+2 for color,_ in coloring]

    print("colors:\n", colors)

    #pos = nx.spring_layout(G)
    #nx.draw(G, pos, with_labels=True)
    #nx.draw_networkx_edge_labels(G, pos)

    nx.draw_circular(G, with_labels=True, node_color=color_map)
    plt.show()

            



    



