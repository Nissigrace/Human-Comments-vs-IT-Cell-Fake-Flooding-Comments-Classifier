"""
Filename: src/models/multiple_linear_regression.py
Purpose: Implement Multiple Linear Regression using several numeric features.
Syllabus connection: Unit 2 - Multiple Linear Regression.
Explanation:
- Extends Simple Linear Regression to model predictions using multiple predictor variables:
  y = w1*x1 + w2*x2 + w3*x3 + ... + b
  - y = engagement_rate (Target variable)
  - x1 = posting_frequency_per_min, x2 = account_age_days, x3 = similarity_to_trending (Features)
  - w1, w2, w3 = coefficients (slopes)
  - b = y-intercept (Bias)
- Demonstrates how multiple indicators combine to predict numerical outcomes.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_multiple_linear_regression(X_train, y_train):
    """
    Fits a Multiple Linear Regression model using Scikit-Learn.
    X_train should be a 2D matrix of shape (n_samples, n_features).
    """
    # 1. Initialize model
    model = LinearRegression()
    
    # 2. Fit to data
    model.fit(X_train, y_train)
    
    # 3. Retrieve mathematical parameters
    coefficients = model.coef_
    intercept = model.intercept_
    
    return model, coefficients, intercept

def evaluate_multiple_regression(model, X_test, y_test):
    """
    Evaluates Multiple Regression using MSE and R2.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Adjusted R2 accounts for the number of features in the model.
    # Formula: 1 - [(1 - R2) * (n - 1) / (n - k - 1)]
    # Where n = sample size, k = number of predictors
    n = len(y_test)
    k = X_test.shape[1]
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
    
    return {
        "MSE": round(mse, 4),
        "R2_Score": round(r2, 4),
        "Adjusted_R2": round(adjusted_r2, 4)
    }

if __name__ == "__main__":
    print("Testing Multiple Linear Regression...")
    X_dummy = np.array([
        [1, 50, 0.1],
        [2, 100, 0.2],
        [3, 150, 0.4],
        [4, 200, 0.7],
        [5, 250, 0.9]
    ])
    y_dummy = np.array([12.5, 24.8, 37.2, 49.1, 61.5])
    
    model, coeffs, intercept = train_multiple_linear_regression(X_dummy, y_dummy)
    print("Coefficients (weights):", coeffs)
    print("Intercept (bias):", intercept)
    
    metrics = evaluate_multiple_regression(model, X_dummy, y_dummy)
    print("Evaluation metrics:", metrics)
