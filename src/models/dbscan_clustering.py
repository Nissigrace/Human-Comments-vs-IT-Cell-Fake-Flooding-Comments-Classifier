"""
Filename: src/models/dbscan_clustering.py
Purpose: Implement DBSCAN density-based clustering for outlier/spam comment group detection.
Syllabus connection: Unit 7 - DBSCAN.
Explanation:
- DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise.
- It finds clusters based on the density of data points in space, and does not require pre-specifying the number of clusters.
- Parameters:
  - eps (Epsilon): Maximum distance search radius around a point.
  - min_samples: Minimum number of points in the neighborhood of a point to classify it as a core point.
- Types of Points:
  1. Core Points: Have >= min_samples points within their eps neighborhood.
  2. Border Points: Have < min_samples points in their eps neighborhood, but lie within the eps neighborhood of a core point.
  3. Noise / Outlier Points: Any point that is neither a core point nor a border point. Marked as -1 in Scikit-Learn.
- Extremely useful for spam/bot detection, because irregular/outlier bot activity will naturally fall into noise (-1).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

def fit_dbscan(X, eps=0.5, min_samples=5):
    """
    Fits DBSCAN clustering algorithm.
    """
    # 1. Initialize DBSCAN model
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    
    # 2. Fit and predict labels
    labels = dbscan.fit_predict(X)
    
    # Identify unique clusters (excluding noise label -1)
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = list(labels).count(-1)
    
    # Compute Silhouette Score (only if we have at least 1 cluster and some clustered points)
    # Silhouette score is not defined if there's only noise or only 1 cluster.
    try:
        if n_clusters > 1:
            # We calculate silhouette only for clustered points, or on all points
            sil_score = silhouette_score(X, labels)
        else:
            sil_score = 0.0
    except Exception:
        sil_score = 0.0
        
    return dbscan, labels, n_clusters, n_noise, round(sil_score, 4)

if __name__ == "__main__":
    print("Testing DBSCAN...")
    # 8 points: a dense group of 6, and 2 far away outliers
    X_dummy = np.array([
        [1.0, 1.0], [1.1, 1.0], [1.0, 1.1], [1.2, 0.9], [1.0, 0.9], [0.9, 1.0],
        [8.0, 8.0], [9.0, 9.0]
    ])
    
    model, labels, n_cl, n_no, sil = fit_dbscan(X_dummy, eps=0.5, min_samples=3)
    print("Labels (-1 represents noise/outliers):", labels)
    print("Number of clusters:", n_cl)
    print("Number of noise points:", n_no)
    print("Silhouette Score:", sil)
