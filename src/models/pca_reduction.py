"""
Filename: src/models/pca_reduction.py
Syllabus connection: Unit 8 - PCA (Principal Component Analysis).
Purpose: Implement Principal Component Analysis for dimensionality reduction of text features.
Explanation:
- PCA is an unsupervised linear transformation technique used for dimensionality reduction.
- It identifies orthogonal axes (principal components) that maximize the variance of the data.
- Mathematical steps:
  1. Standardize/Center the data.
  2. Compute the Covariance Matrix.
  3. Calculate the Eigenvalues and Eigenvectors of the covariance matrix.
  4. Sort eigenvalues in descending order to rank eigenvectors (components).
  5. Project the original data onto the top eigenvectors.
- Explained Variance Ratio: The proportion of the dataset's variance that lies along the axis of each principal component.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def fit_pca(X, n_components=2):
    """
    Fits PCA and transforms high-dimensional features matrix X to reduced coordinates.
    """
    # 1. Initialize PCA model
    pca = PCA(n_components=n_components, random_state=42)
    
    # 2. Fit and transform the inputs
    X_reduced = pca.fit_transform(X)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    total_var = np.sum(explained_variance)
    
    print(f"[PCA] Explained variance by component: {explained_variance}")
    print(f"[PCA] Total explained variance ({n_components} components): {total_var:.4f}")
    
    return pca, X_reduced, total_var

if __name__ == "__main__":
    print("Testing PCA module...")
    # Mock data with 5 samples and 4 dimensions
    X_dummy = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [1.2, 2.2, 3.1, 4.2],
        [5.0, 6.0, 7.0, 8.0],
        [5.1, 6.1, 7.2, 8.1],
        [10.0, 11.0, 12.0, 13.0]
    ])
    
    pca_model, reduced, total_var = fit_pca(X_dummy, n_components=2)
    print("Reduced shape (should be 5, 2):", reduced.shape)
