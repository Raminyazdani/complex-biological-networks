"""
Network Controllability Analysis Module

Analyzes controllability properties of complex networks using control theory.
Implements exact and structural controllability tests for directed networks.

Features:
- Controllability matrix computation
- Exact controllability testing (rank-based)
- Structural controllability analysis
- Network topology visualization with control inputs
- Support for cactus graph detection

Key Concepts:
- Controllability: ability to drive a network to any desired state using control inputs
- Driver nodes: minimal set of nodes that must be controlled
"""

from itertools import combinations
from colorama import Fore, Style, init
import array_to_latex as a2l

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nbconvert.exporters import latex
from scipy.linalg import block_diag
from numpy.linalg import matrix_rank
from scipy.linalg import block_diag, lu


def OUTPUT(key,arr, transpose=False, decimals=8):
    # 2D array is not need [1] so we should check the dimension
    if np.asarray(arr).ndim == 1:
        # 1D array can't get transpose if we not add '[]' at before and after array's variable. Reference: [1]
        arr = np.asarray([arr])
    else:
        # 2D array is fine, not like [1]
        arr = np.asarray(arr)

    if transpose:
        # thanks to [1] so we can also transpose 1D, not just 2D
        arr = arr.T

    # round the number to @param decimals
    arr = np.around(arr, decimals=decimals)

    row_num = arr.shape[0]
    col_num = arr.shape[1]

    if (col_num == 1):
        result = '[■('
        result += '@'.join([str(arr[row][0]) for row in range(row_num)])
        result += ')]'
    elif (row_num == 1):
        result = '[■('
        result += '&'.join([str(arr[0][col]) for col in range(col_num)])
        result += ')]'
    else:
        result = '[■('
        result += '@'.join(['&'.join([str(arr[row, col]) for col in range(col_num)]) for row in range(row_num)])
        result += ')]'
    rank = key.split(" rank=")[-1]
    name=key.split(" rank")[0]
    print(name+" = "+result+fr",\ \rho({name})={rank}")
def plot_system_graph(A_plot, B, vertices, Input_names):
    A_plot = A_plot.transpose()
    n = A_plot.shape[0]
    G = nx.DiGraph()

    # Add nodes to the graph (state vertices)
    G.add_nodes_from(vertices)

    # Add edges corresponding to the adjacency matrix A_plot (state transitions)
    for i in range(n):
        for j in range(n):
            if A_plot[i, j] != 0:
                G.add_edge(vertices[i], vertices[j], weight=A_plot[i, j])

    # Add edges for the control inputs B (input to state connections)
    inps = []
    if len(B) > 0:
        for j, input_name in enumerate(Input_names):
            inps.append(input_name)
            for i in range(n):
                if B[i, j] != 0:  # Check each column in B
                    G.add_edge(input_name, vertices[i], weight=B[i, j], color='#FF0000', style='dashed')

    # Position the graph
    try:
        pos = nx.bfs_layout(G, start=inps)
    except:
        pos = nx.bfs_layout(G, start=vertices[0])
    # Extract edge labels, colors, and styles
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_colors = [G[u][v].get('color', 'black') for u, v in G.edges()]
    edge_styles = [G[u][v].get('style', 'solid') for u, v in G.edges()]

    # Draw the nodes (state and input) and labels
    nx.draw(G, pos, node_size=2000, node_color=["skyblue" if node in vertices else "green" for node in G.nodes()],
            font_size=15, font_weight="bold", with_labels=True, arrows=True, arrowstyle='-|>', arrowsize=20)

    # Draw edges with styles

    for edge, color, style in zip(G.edges(), edge_colors, edge_styles):
        nx.draw_networkx_edges(G, pos, edgelist=[edge], style=style, edge_color=color, width=3)

    # Draw edge labels with white rectangle background
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black',
                                 bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))

    plt.title('System State Graph with Control Inputs')
    plt.show()


def controllability_matrix(A, B, printable,vertic=False):
    n = A.shape[0]
    controllability_mat = B
    prints = {f"A rank={np.linalg.matrix_rank(A)}": A, f"B rank={np.linalg.matrix_rank(B)}": B,
              f"AB rank={np.linalg.matrix_rank(np.hstack((A, B)))}": np.hstack((A, B))}
    keysss = []
    for i in range(1, n):
        A_power_B = np.linalg.matrix_power(A, i) @ B
        prints[f"A^{i} rank={np.linalg.matrix_rank(np.linalg.matrix_power(A, i))}"] = np.linalg.matrix_power(A, i)
        prints[f"A^{i} B rank={np.linalg.matrix_rank(A_power_B)}"] = A_power_B
        controllability_mat = np.hstack((controllability_mat, A_power_B))
        keysss.append(f"A^{i} B")

    prints[f"C[B {" ".join(keysss)}] rank={np.linalg.matrix_rank(controllability_mat)}"] = controllability_mat
    if printable:
        if vertic == True:
            for key,val in prints.items():
                OUTPUT(key=key,arr=val)
                print()
            return controllability_mat
        # Convert each matrix to a string representation
        rows = []
        for i in range(n):  # iterate over rows
            row_parts = []
            for key in prints:
                row_parts.append(' '.join([f"{item: .2f}" for item in prints[key][i]]))
            rows.append('   |   '.join(row_parts))

        # Print each row in a single line
        header = ""
        cols = []
        headers_temp = [x for x in prints.keys()]
        cols = [len(x) + 1 for x in rows[0].split("|")]
        cols = [" " * x for x in cols]
        for c in cols:
            t = headers_temp.pop(0) + c
            t = t[:len(c)]
            header += t
        print(header)
        for row in rows:
            for i in row:
                try:
                    if float(i) != 0:
                        print(Fore.BLUE + i + Style.RESET_ALL, end="")
                    else:
                        print(i, end="")
                except:
                    print(i, end="")
            print()
    return controllability_mat


def is_controllable(A, B, ctrl_mat):
    rank = np.linalg.matrix_rank(ctrl_mat)
    # print(f"Rank of Controllability Matrix: {rank}")
    # print(f"Number of States: {A.shape[0]}")
    return rank == A.shape[0]


def generate_A(vertices, edges_weight):
    n = len(vertices)
    vertex_index = {vertex: i for i, vertex in enumerate(vertices)}

    # Initialize adjacency matrix A_plot
    A = np.zeros((n, n), dtype=float)

    for edge, weight in edges_weight.items():
        v1, v2 = edge.split('-')
        i = vertex_index[v1]
        k = vertex_index[v2]

        # Update adjacency matrix with the weight of the edge
        A[i, k] = weight
    A = A.transpose()
    return A


def is_simple_directed_cycle(G):
    """
    Check if a subgraph G is a simple directed cycle.
    A_plot simple directed cycle should have:
    - The same number of edges as nodes
    - Each node should have in-degree and out-degree equal to 1
    """
    if len(G) < 2:  # A_plot cycle requires at least 2 nodes
        return False

    for node in G.nodes():
        if G.in_degree(node) != 1 or G.out_degree(node) != 1:
            return False

    return len(G.edges()) == len(G.nodes())


def is_directed_cactus(G):
    """
    Check if a directed graph is a cactus.
    A_plot directed graph is a cactus if every strongly connected component is either:
    - A_plot single vertex, or
    - A_plot simple directed cycle.
    """
    # Find all strongly connected components (SCCs)
    sccs = list(nx.strongly_connected_components(G))

    for scc in sccs:
        if len(scc) == 1:
            continue  # Single node component is fine
        subgraph = G.subgraph(scc)
        if not is_simple_directed_cycle(subgraph):
            return False
    return True


def is_structurally_controllable(A, B):
    N = A.shape[0]

    # Step 1: Form the structured matrix [A_plot^T; B]
    structured_matrix = np.hstack((A, B))

    # Step 2: Check if the structured matrix [A_plot^T; B] is irreducible
    # Instead of LU, you can implement a more direct graph-based irreducibility check
    # (Skipping this step as it requires specific graph-theoretic methods)

    # Step 3: Check if the generic rank of [A_plot^T; B] is N
    rank_of_structured_matrix = np.linalg.matrix_rank(structured_matrix)
    if rank_of_structured_matrix < N:
        # print(f"The generic rank of [A; B] is less than {N}. The system is not structurally controllable.")
        return False

    # If the rank is sufficient, assume structural controllability
    # print("The system is structurally controllable.")
    return True


def is_structurally_controllable_2(A, B):
    # Get dimensions of A_plot and B
    n, m = B.shape

    # Initialize controllability matrix
    controllability_matrix = B

    # Compute controllability matrix: [B, AB, A_plot^2B, ..., A_plot^(n-1)B]
    for i in range(1, n):
        controllability_matrix = np.hstack((controllability_matrix, np.dot(np.linalg.matrix_power(A, i), B)))

    # Check if the controllability matrix has full rank
    if matrix_rank(controllability_matrix) == n:
        return True
    else:
        return False




# Adjust the input matrix B accordingly if needed

def generate_B(vertices, Inputs):
    n = len(vertices)
    max_cols = len(Inputs)  # The number of inputs will determine the number of columns

    B = np.zeros((n, max_cols), dtype=float)  # Initialize B with zeros

    vertex_index = {vertex: i for i, vertex in enumerate(vertices)}

    for col_idx, (input_name, input_edges) in enumerate(Inputs.items()):
        for vertex, weight in input_edges.items():
            row_idx = vertex_index[vertex]  # Find the corresponding row index for the vertex
            B[row_idx, col_idx] = weight  # Set the weight in the B matrix

    return B


if __name__ == '__main__':

    # Adjust the lists for each specific scenario
    lists = []

    # Case 1: exact controllable True, structural controllable False
    # Requires A_plot and B such that the controllability matrix has full rank but the graph is not strongly connected
    lists.append(
        (["X1", "X2", "X3", "X4"], {"X1-X4": 1, "X1-X3": 1, "X4-X3": 1, "X1-X2": 1}, {"b1": {"X1": 1}, "b2": {"X2": 1}}))

    lists.append(
        (["X1", "X2", "X3"], {"X1-X2": 1, "X2-X3": 1}, {"b1": {"X1": 1}}))

    for idx, (vertices, edges_weight, Inputs) in enumerate(lists):
        A = generate_A(vertices, edges_weight)
        B = generate_B(vertices, Inputs)

        plot_system_graph(A, B, vertices, Inputs.keys())

        ctrl_matrix = controllability_matrix(A, B, printable=True)
        exact = is_controllable(A, B, ctrl_matrix)
        is_structurally_controllable_ = is_structurally_controllable(A, B)

        print(f"Graph {idx + 1} - Exact Controllable: {exact} , structural controllable: {is_structurally_controllable_}")

        input("continue ? ")
