"""
Minimum Driver Node Finder

Determines the minimal set of driver nodes required for complete network controllability.
Exhaustively searches all possible control configurations to find optimal solutions.

Features:
- Exhaustive search over all node subsets
- Identification of minimum driver node count
- Optimization for minimum input connections
- Visualization of optimal control configurations

Note: Computational complexity grows exponentially with network size.
Best suited for small to medium networks (< 10 nodes).
"""

from itertools import combinations

from main3 import generate_A, plot_system_graph, generate_B, controllability_matrix, is_controllable, \
    is_structurally_controllable

if __name__ == '__main__':

    data = (["1", "2", "3", "4"], {"1-2": 1, "2-3": 1, "2-4": 1})
    vertices = data[0]
    edges_weight = data[1]
    A = generate_A(vertices, edges_weight)
    B = {}

    plot_system_graph(A, B, vertices, edges_weight)
    print()
    n = len(A)
    node_sets = []
    for r in range(1, n + 1):
        for node_set in combinations(vertices, r):
            node_sets.append(node_set)
    f_nodes = []
    for r in range(1, len(node_sets) + 1):
        for n in combinations(node_sets, r):
            f_nodes.append(n)
    f_nodes_final = []

    for n in f_nodes:
        temp = {}
        for idx, item in enumerate(n):
            temp[f"u{idx + 1}"] = {}
            for m in item:
                temp[f"u{idx + 1}"][m] = 1
        f_nodes_final.append(temp)

    good_nodes = []
    for idx, item in enumerate(f_nodes_final):
        print(idx, "/", len(f_nodes_final), end="\r")
        B = generate_B(vertices, item)
        ctrl_matrix = controllability_matrix(A, B, printable=False)
        exact = is_controllable(A, B, ctrl_matrix)

        if exact == True:
            good_nodes.append(item)
    print()
    print(len(f_nodes_final))
    print(len(good_nodes))
    minimum_driver = None

    for g in good_nodes:
        if minimum_driver == None:
            minimum_driver = len(list(g.keys()))
            continue
        if len(list(g.keys())) < minimum_driver:
            minimum_driver = len(list(g.keys()))
    c_driver = 0
    drivers = []
    for g in good_nodes:
        if len(list(g.keys())) == minimum_driver:
            c_driver += 1
            drivers.append(g)

    minimum_ins_driver = None
    dirvers_final = []

    for g in drivers:
        t = 0

        for key, value in g.items():
            t += len(value)

        if minimum_ins_driver == None:
            minimum_ins_driver = t
            continue
        if t < minimum_ins_driver:
            minimum_ins_driver = t

    for g in drivers:
        t = 0
        for key, value in g.items():
            t += len(value)
        if t <= minimum_ins_driver:
            dirvers_final.append(g)

    for idx,d in enumerate(dirvers_final):
        B = generate_B(vertices,d)
        ctrl_matrix = controllability_matrix(A, B, printable=True,vertic=True)
        exact = is_controllable(A, B, ctrl_matrix)
        is_structurally_controllable_ = is_structurally_controllable(A, B)
        plot_system_graph(A,B,vertices,d.keys())
        print(
            f"Graph {idx + 1} - Exact Controllable: {exact} , structural controllable: {is_structurally_controllable_}")
