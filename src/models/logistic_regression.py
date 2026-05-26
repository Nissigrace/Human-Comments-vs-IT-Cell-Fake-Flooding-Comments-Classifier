"""
Filename: src/models/logistic_regression.py
Purpose: Implement Logistic Regression for binary classification.
Syllabus connection: Unit 3 - Logistic Regression.
Explanation:
- Logistic Regression models the probability that an observation belongs to a particular class (binary classification).
- Sigmoid activation function: g(z) = 1 / (1 + e^-z) maps any real value z to a probability between 0 and 1.
- Hypothesis: h_w(x) = g(w^T * x)
- Cost Function (Binary Cross-Entropy / Log Loss): 
  J(w) = -1/n * sum( y*log(h_w(x)) + (1-y)*log(1 - h_w(x)) )
- Optimization: Gradient Descent updates weights to minimize loss.
- Estimation: Maximum Likelihood Estimation (MLE) finds parameters that maximize the probability of observing the actual data labels.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc

def train_logistic_regression(X_train, y_train):
    """
    Fits a Logistic Regression classifier using Scikit-Learn.
    """
    # 1. Initialize classifier
    model = LogisticRegression(random_state=42)
    
    # 2. Fit to data
    model.fit(X_train, y_train)
    
    # 3. Retrieve mathematical parameters
    weights = model.coef_[0]
    intercept = model.intercept_[0]
    
    return model, weights, intercept

def get_predictions_with_threshold(model, X, threshold=0.5):
    """
    Predicts labels based on a custom decision threshold (instead of standard 0.5).
    - If P(Class 1) >= threshold: predict 1
    - Else: predict 0
    This is highly useful for analyzing accuracy, sensitivity (recall), and specificity trade-offs.
    """
    # Predict probabilities for class 1 (index 1)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Apply custom threshold
    predictions = (probabilities >= threshold).astype(int)
    
    return predictions, probabilities

def evaluate_logistic_model(model, X_test, y_test, threshold=0.5):
    """
    Calculates key classification metrics.
    """
    predictions, probabilities = get_predictions_with_threshold(model, X_test, threshold)
    
    # Calculate ROC Curve and Area Under Curve (AUC)
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    roc_auc = auc(fpr, tpr)
    
    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": round(roc_auc, 4)
    }

if __name__ == "__main__":
    print("Testing Logistic Regression...")
    X_dummy = np.array([
        [0.1, 0.9],
        [0.2, 0.8],
        [0.8, 0.2],
        [0.9, 0.1]
    ])
    y_dummy = np.array([0, 0, 1, 1])
    
    model, w, b = train_logistic_regression(X_dummy, y_dummy)
    print("Weights (coefficients):", w)
    print("Intercept (bias):", b)
    
    preds, probs = get_predictions_with_threshold(model, X_dummy, threshold=0.6)
    print("Predictions with threshold 0.6:", preds)
    print("Probabilities:", probs)
