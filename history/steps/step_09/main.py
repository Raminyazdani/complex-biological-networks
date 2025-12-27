"""
Clustering Analysis Module

Performs K-means and hierarchical clustering analysis on network data.
Generates distance matrices, dendrograms, and heatmap visualizations.

Features:
- K-means clustering (3 clusters by default)
- Hierarchical clustering using Ward's method
- Euclidean distance matrix computation
- Visualization with matplotlib and seaborn
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans

A = np.array([1, 7, 10, 11, 14, 20]).reshape(-1, 1)

distance_matrix = np.zeros((len(A), len(A)))

for i in range(len(A)):
    for j in range(len(A)):
        distance_matrix[i][j] = np.sqrt((A[i][0] - A[j][0]) ** 2)

linkage_matrix = linkage(A, method='ward')

kmeans = KMeans(n_clusters=3, random_state=0).fit(A)

labels = kmeans.labels_

print("\nK-means Clustering Results:")
for i in range(3):
    cluster_elements = A[labels == i].flatten()
    print(f'Cluster {i+1}: {cluster_elements}')


plt.figure(figsize=(8, 6))
sns.heatmap(distance_matrix, annot=True, cmap="Blues_r", xticklabels=A.flatten(), yticklabels=A.flatten())
plt.title("Euclidean Distance Heatmap (Inverted Blue Shades)")
plt.xlabel("Elements of Set A_plot")
plt.ylabel("Elements of Set A_plot")
plt.show()

plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, labels=A.flatten(), orientation='top', color_threshold=7)
plt.title("Dendrogram (Hierarchical Clustering Tree Map)")
plt.xlabel("Elements of Set A_plot")
plt.ylabel("Distance")
plt.show()

plt.figure(figsize=(8, 6))
for i in range(3):
    cluster_points = A[labels == i]
    plt.scatter(cluster_points, np.zeros_like(cluster_points), label=f'Cluster {i+1}')
    for point in cluster_points:
        plt.annotate(int(point[0]), (point, 0), textcoords="offset points", xytext=(0, 10), ha='center')

plt.xlabel('Elements of Set A_plot')
plt.title('K-means Clustering of Set A_plot')
plt.legend()
plt.grid(True)
plt.show()
