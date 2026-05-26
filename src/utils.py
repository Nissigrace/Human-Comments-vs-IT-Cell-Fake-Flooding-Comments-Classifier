"""
Filename: src/utils.py
Purpose: Utility helper functions for clustering tendency (Hopkins statistic), distance calculations,
         confusion matrix evaluation, and manual weight updates (for syllabus practicals).
Explanation:
- Implements Euclidean & Manhattan distance calculations.
- Implements the Hopkins statistic for clustering tendency evaluation.
- Implements classification evaluation metrics from scratch (Accuracy, Sensitivity, Specificity, Precision, Recall, F1-Score).
- Written with thorough explanations and clean code for educational purposes.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# ==========================================
# 1. DISTANCE METRIC FUNCTIONS
# ==========================================

def calculate_euclidean_distance(point1, point2):
    """
    Computes Euclidean (L2) distance between two points.
    Formula: sqrt(sum((p1_i - p2_i)^2))
    Represents the straight-line distance between two points in Euclidean space.
    """
    p1 = np.array(point1)
    p2 = np.array(point2)
    squared_diff = (p1 - p2) ** 2
    sum_squared = np.sum(squared_diff)
    distance = np.sqrt(sum_squared)
    return distance

def calculate_manhattan_distance(point1, point2):
    """
    Computes Manhattan (L1 / Taxicab / City Block) distance between two points.
    Formula: sum(|p1_i - p2_i|)
    Represents the distance measured along axes at right angles.
    """
    p1 = np.array(point1)
    p2 = np.array(point2)
    abs_diff = np.abs(p1 - p2)
    distance = np.sum(abs_diff)
    return distance


# ==========================================
# 2. CLUSTERING TENDENCY: HOPKINS STATISTIC
# ==========================================

def calculate_hopkins_statistic(X, m=None):
    """
    Computes the Hopkins statistic to assess clustering tendency of dataset X.
    Hopkins Statistic (H) measures the spatial randomness of points.
    
    Formula: H = sum(u_i) / (sum(u_i) + sum(v_i))
    Where:
    - u_i: Distance of synthetic random points (drawn uniformly from X's bounding box) to their nearest neighbor in X.
    - v_i: Distance of real points chosen randomly from X to their nearest neighbor in X (excluding itself).
    
    Interpretation:
    - H ~ 0.5: Randomly distributed (no clustering tendency).
    - H -> 1.0: Highly clustered (strong clustering tendency).
    - H -> 0.0: Uniformly spaced grid (regular distribution).
    """
    X = np.array(X)
    n = X.shape[0]
    d = X.shape[1]
    
    if m is None:
        m = int(0.1 * n)  # Sample size, usually 10% of dataset
        if m < 1:
            m = 1
            
    # Fit nearest neighbors model on the actual dataset
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    
    # Generate m synthetic random points uniformly distributed over the bounding box of X
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    synthetic_points = np.random.uniform(low=mins, high=maxs, size=(m, d))
    
    # Calculate u_i: distance from synthetic points to nearest real point in X
    u_distances, _ = nbrs.kneighbors(synthetic_points, n_neighbors=1)
    sum_u = np.sum(u_distances)
    
    # Select m random real points from actual dataset X
    random_indices = np.random.choice(n, size=m, replace=False)
    real_sample_points = X[random_indices]
    
    # Calculate v_i: distance from real sample points to their nearest neighbor in X
    # Note: we find 2 nearest neighbors because the first nearest neighbor of a point in X is the point itself (distance = 0)
    real_distances, _ = nbrs.kneighbors(real_sample_points, n_neighbors=2)
    # Use the 2nd neighbor distance (index 1)
    sum_v = np.sum(real_distances[:, 1])
    
    # Compute Hopkins statistic
    hopkins_val = sum_u / (sum_u + sum_v)
    return hopkins_val


# ==========================================
# 3. CLASSIFICATION EVALUATION METRICS
# ==========================================

def calculate_confusion_matrix_metrics(y_true, y_pred):
    """
    Computes Accuracy, Sensitivity, Specificity, Precision, Recall, and F1-Score from scratch.
    
    Definitions:
    - True Positive (TP): Actual 1, Predicted 1
    - True Negative (TN): Actual 0, Predicted 0
    - False Positive (FP): Actual 0, Predicted 1 (Type I error)
    - False Negative (FN): Actual 1, Predicted 0 (Type II error)
    
    Formulas:
    - Accuracy = (TP + TN) / (TP + TN + FP + FN)
    - Sensitivity (Recall) = TP / (TP + FN)
    - Specificity = TN / (TN + FP)
    - Precision = TP / (TP + FP)
    - F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Counts
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate metrics, handling division-by-zero checks
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    recall = sensitivity  # Recall is mathematically identical to sensitivity
    
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        "TP": int(TP),
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
        "Accuracy": round(accuracy, 4),
        "Sensitivity (Recall)": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "Precision": round(precision, 4),
        "F1-Score": round(f1_score, 4)
    }
    
    return metrics


# ==========================================
# 4. ITERATIVE SOLVERS (FOR PRACTICALS)
# ==========================================

def solve_gradient_descent_linear(x, y, m_start=0.0, c_start=0.0, alpha=0.01, iterations=5):
    """
    Simulates simple linear regression gradient descent step-by-step.
    Equation: y_pred = m * x + c
    Loss (MSE): J(m,c) = (1/2n) * sum((y_pred - y)^2)
    Partial Derivatives:
      dJ/dm = (1/n) * sum((y_pred - y) * x)
      dJ/dc = (1/n) * sum(y_pred - y)
    Parameter updates:
      m = m - alpha * (dJ/dm)
      c = c - alpha * (dJ/dc)
    """
    n = len(x)
    x = np.array(x)
    y = np.array(y)
    
    m = m_start
    c = c_start
    
    history = []
    
    for epoch in range(1, iterations + 1):
        # 1. Predictions
        y_pred = m * x + c
        # 2. Errors
        error = y_pred - y
        # 3. Cost (MSE)
        cost = np.mean(error ** 2) / 2
        # 4. Gradients
        dm = np.mean(error * x)
        dc = np.mean(error)
        # 5. Parameter update
        m_new = m - alpha * dm
        c_new = c - alpha * dc
        
        history.append({
            "Epoch": epoch,
            "m_old": round(m, 4),
            "c_old": round(c, 4),
            "Cost": round(cost, 4),
            "dm": round(dm, 4),
            "dc": round(dc, 4),
            "m_new": round(m_new, 4),
            "c_new": round(c_new, 4)
        })
        
        m = m_new
        c = c_new
        
    return history


def solve_logistic_weight_update(x, y, w_start, alpha=0.1, iterations=1):
    """
    Simulates gradient descent for logistic regression weight updates.
    Hypothesis: h_w(x) = sigmoid(w^T * x)
    Sigmoid: g(z) = 1 / (1 + e^-z)
    Update rule: w_j = w_j + alpha * (y - h_w(x)) * x_j
    (Assuming Gradient Ascent for maximizing Likelihood, or w_j = w_j - alpha * (h_w(x) - y) * x_j for minimizing loss)
    """
    # Initialize variables
    w = np.array(w_start, dtype=float)
    x = np.array(x, dtype=float)
    y = float(y)
    
    history = []
    
    for epoch in range(1, iterations + 1):
        # Dot product
        z = np.dot(w, x)
        # Prediction (probability)
        h = 1 / (1 + np.exp(-z))
        # Error (actual - predicted)
        err = y - h
        # Weight updates
        w_new = w + alpha * err * x
        
        history.append({
            "Epoch": epoch,
            "w_old": w.copy(),
            "z": round(z, 4),
            "h_w(x)": round(h, 4),
            "Error (y - h)": round(err, 4),
            "w_new": w_new.copy()
        })
        w = w_new
        
    return history


if __name__ == "__main__":
    print("Testing Utils...")
    p1 = [1, 2]
    p2 = [4, 6]
    print("Euclidean distance:", calculate_euclidean_distance(p1, p2))
    print("Manhattan distance:", calculate_manhattan_distance(p1, p2))
    
    y_true = [1, 0, 1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 1, 1, 0, 1]
    print("Confusion Matrix metrics:", calculate_confusion_matrix_metrics(y_true, y_pred))
    
    # Test Hopkins on noise
    X = np.random.normal(size=(100, 2))
    print("Hopkins Statistic (Random noise):", calculate_hopkins_statistic(X))
