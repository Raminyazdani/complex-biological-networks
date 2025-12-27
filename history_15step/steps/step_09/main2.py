"""
Causal Discovery Module

Implements the PC (Peter-Clark) algorithm for inferring causal relationships
from observational data using conditional independence testing.

Features:
- PC algorithm for causal structure learning
- Partial correlation-based conditional independence tests
- Edge orientation using statistical inference
- Network visualization of causal graphs

Reference: PC algorithm by Peter Spirtes and Clark Glymour
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import itertools

np.random.seed(40)
n_samples = 100

Z = np.random.normal(size=n_samples)
T = np.random.normal(size=n_samples)

X = 0.5 * Z + np.random.normal(size=n_samples)
W = 0.3 * T + np.random.normal(size=n_samples)
Y = 0.4 * W + 0.5 * Z + np.random.normal(size=n_samples)

data = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z, 'W': W, 'T': T})
data.to_csv("random_pc_data.csv")

def partial_correlation(data, x, y, control):
    if len(control) == 0:
        return pearsonr(data[x], data[y])[1]
    lr_x = LinearRegression().fit(data[control], data[x])
    lr_y = LinearRegression().fit(data[control], data[y])
    residuals_x = data[x] - lr_x.predict(data[control])
    residuals_y = data[y] - lr_y.predict(data[control])
    return pearsonr(residuals_x, residuals_y)[1]

def conditional_independence_test(data, var1, var2, conditioning_set, alpha=0.05):
    p_value = partial_correlation(data, var1, var2, list(conditioning_set))
    print(f"CI Test: {var1} _|_ {var2} | {conditioning_set}, p-value: {p_value}")
    return p_value > alpha

variables = data.columns.tolist()
edges = [(var1, var2) for i, var1 in enumerate(variables) for var2 in variables[i + 1:]]
steps = [edges.copy()]

max_cardinality = len(variables) - 2
for k in range(max_cardinality + 1):
    for (var1, var2) in edges.copy():
        for conditioning_set in itertools.combinations(set(variables) - {var1, var2}, k):
            if conditional_independence_test(data, var1, var2, conditioning_set):
                edges.remove((var1, var2))
                break
    steps.append(edges.copy())

def orient_edges(edges, data, alpha=0.001):
    directed_edges = []
    for (var1, var2) in edges:
        p_value1 = partial_correlation(data, var1, var2, [])
        p_value2 = partial_correlation(data, var2, var1, [])
        if p_value1 <= p_value2:
            directed_edges.append((var2, var1))
        else:
            directed_edges.append((var1, var2))
    return directed_edges

directed_edges = orient_edges(edges, data)
steps.append(directed_edges)

def draw_graph(edges, title, directed=False):
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_edges_from(edges)
    pos = nx.circular_layout(G)
    plt.figure()
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=16, font_weight='bold', arrows=directed)
    plt.title(title)
    plt.show()

titles = [f"Step {i+1}" for i in range(len(steps))]
for i, edges in enumerate(steps):
    draw_graph(edges, titles[i], directed=(i == len(steps) - 1))
print(*steps, sep="\n")
