"""
Filename: src/models/hierarchical_clustering.py
Purpose: Implement Agglomerative Hierarchical Clustering.
Syllabus connection: Unit 6 - Hierarchical Clustering.
Explanation:
- Hierarchical Clustering builds a tree of clusters (dendrogram).
- Agglomerative is a bottom-up approach:
  1. Start with each point as an individual cluster.
  2. Compute distance matrix between all clusters.
  3. Find the two closest clusters and merge them into a single cluster.
  4. Recompute distance between the new cluster and all other clusters using a Linkage criterion:
     - Ward Linkage: Minimizes the variance of clusters being merged (default).
     - Complete Linkage: Distance is defined by the maximum distance between points in the two clusters.
     - Single Linkage: Distance is defined by the minimum distance between points in the two clusters.
     - Average Linkage: Distance is defined by the average distance between all pairs of points.
  5. Repeat steps 3 & 4 until all points are merged into a single root cluster.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def fit_hierarchical_clustering(X, n_clusters=2, linkage_type='ward'):
    """
    Fits Agglomerative Clustering and returns model and labels.
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
    labels = model.fit_predict(X)
    
    try:
        sil_score = silhouette_score(X, labels)
    except Exception:
        sil_score = 0.0
        
    return model, labels, round(sil_score, 4)

def compute_linkage_matrix(X, linkage_type='ward'):
    """
    Computes Scipy linkage matrix needed to plot a dendrogram.
    The linkage matrix Z contains information about which clusters were merged at each step.
    Z has shape (n-1, 4): Z[i, 0] and Z[i, 1] are merged clusters, Z[i, 2] is distance, Z[i, 3] is cluster size.
    """
    Z = linkage(X, method=linkage_type)
    return Z

if __name__ == "__main__":
    print("Testing Hierarchical Clustering...")
    X_dummy = np.array([
        [1.0, 1.1],
        [1.2, 0.9],
        [8.0, 8.5],
        [8.5, 7.9]
    ])
    
    model, labels, sil = fit_hierarchical_clustering(X_dummy, n_clusters=2)
    print("Labels assigned:", labels)
    print("Silhouette Score:", sil)
    
    Z = compute_linkage_matrix(X_dummy, linkage_type='ward')
    print("Linkage Matrix shape:", Z.shape)
