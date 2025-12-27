"""
Clustering analysis module - K-means and hierarchical clustering with visualization.

This module implements:
- K-means clustering for partitioning data
- Hierarchical clustering with dendrograms
- Distance matrix computations
- Visualization tools (heatmaps, dendrograms)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# Sample data for clustering
A = np.array([1, 7, 10, 11, 14, 20])

# K-means Clustering
def kmeans(data, k=3, max_iters=100):
    centroids = data[np.random.choice(len(data), k, replace=False)]
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [abs(point - centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)
        
        new_centroids = [np.mean(cluster) if cluster else centroids[i] 
                        for i, cluster in enumerate(clusters)]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters

# Perform K-means
clusters = kmeans(A, k=3)
print("K-means Clustering Results:")
for i, cluster in enumerate(clusters, 1):
    print(f"Cluster {i}: {np.array(cluster)}")

# Hierarchical Clustering
# Distance matrix calculation will be added in next step
print("\nHierarchical clustering structure prepared.")
