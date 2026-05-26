"""
Filename: src/models/kmeans_clustering.py
Purpose: Implement K-Means clustering for unsupervised comments grouping.
Syllabus connection: Unit 5 - K-Means Clustering.
Explanation:
- K-Means is a partitioning clustering algorithm that divides n observations into K clusters.
- Objective function (Inertia/WCSS): Minimizes Within-Cluster Sum of Squares:
  J = sum_i( sum_j( ||x_i - mu_j||^2 ) ) for points in cluster j.
- Steps:
  1. Initialize K random centroids.
  2. Assign each data point to the nearest centroid (using Euclidean distance).
  3. Recompute centroids as the mean of all points assigned to that cluster.
  4. Repeat steps 2 & 3 until centroids stabilize (convergence).
- Silhouette Score: Evaluates clustering quality by measuring how close a point is to its cluster members (cohesion)
  vs how far it is from other clusters (separation). Range: [-1, 1].
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def fit_kmeans(X, n_clusters=2):
    """
    Fits K-Means algorithm and returns model, cluster labels, and centroids.
    """
    # 1. Initialize KMeans with clusters, k-means++ for smart initialization of centroids, and random seed
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    
    # 2. Fit the model to the input features (no labels are passed - unsupervised!)
    labels = kmeans.fit_predict(X)
    
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_  # WCSS
    
    # Compute Silhouette Score if possible (requires > 1 cluster and at least n_samples > n_clusters)
    try:
        sil_score = silhouette_score(X, labels)
    except Exception:
        sil_score = 0.0
        
    return kmeans, labels, centroids, inertia, round(sil_score, 4)

def find_optimal_k_elbow(X, max_k=8):
    """
    Computes inertia (WCSS) values for multiple K selections.
    This helps the user plot the 'Elbow curve' to visually find the optimal K.
    """
    inertia_values = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X)
        inertia_values.append(kmeans.inertia_)
        
    return list(k_range), inertia_values

if __name__ == "__main__":
    print("Testing K-Means...")
    X_dummy = np.array([
        [1.0, 1.1],
        [1.2, 0.9],
        [8.0, 8.5],
        [8.5, 7.9]
    ])
    
    model, labels, centers, wcss, sil = fit_kmeans(X_dummy, n_clusters=2)
    print("Labels assigned:", labels)
    print("Centroids:\n", centers)
    print("WCSS (Inertia):", wcss)
    print("Silhouette Score:", sil)
