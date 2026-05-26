"""
Filename: src/models/simple_linear_regression.py
Purpose: Implement Simple Linear Regression to model the relationship between posting frequency and engagement rate.
Syllabus connection: Unit 1 - Simple Linear Regression.
Explanation:
- We fit a model y = mx + c where:
  - y = engagement_rate (Dependent variable)
  - x = posting_frequency_per_min (Independent variable)
  - m = slope (Weight/coefficient)
  - c = y-intercept (Bias)
- Cost Function: Mean Squared Error (MSE) measures the average squared difference between predictions and actual values.
- Gradient Descent: Optimizes parameters (m, c) by taking steps proportional to the negative gradient of the MSE cost.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_simple_linear_regression(X_train, y_train):
    """
    Fits a Simple Linear Regression model using Scikit-Learn.
    """
    # 1. Initialize the model
    # LinearRegression fits a linear model with coefficients w = (w1, ..., wp) 
    # to minimize the residual sum of squares between the observed targets and the predictions.
    model = LinearRegression()
    
    # 2. Fit the model to the training data
    # X_train must be 2D. We reshape it or pass a 2D array.
    model.fit(X_train, y_train)
    
    # 3. Retrieve mathematical parameters
    slope = model.coef_[0]
    intercept = model.intercept_
    
    return model, slope, intercept

def evaluate_regression_model(model, X_test, y_test):
    """
    Evaluates the regression model performance.
    Metrics used:
    - Mean Squared Error (MSE): The average of the squared errors. Lower is better.
    - R-squared (R2 Score): The coefficient of determination. It shows the proportion of 
      variance in the dependent variable that is predictable from the independent variables (scale 0 to 1).
    """
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return {
        "MSE": round(mse, 4),
        "R2_Score": round(r2, 4)
    }

if __name__ == "__main__":
    # Mini test
    print("Testing Simple Linear Regression...")
    X_dummy = np.array([[1], [2], [3], [4], [5]])
    y_dummy = np.array([2.5, 4.8, 7.2, 9.1, 11.5]) # Close to y = 2.2*x + 0.3
    
    model, slope, intercept = train_simple_linear_regression(X_dummy, y_dummy)
    print(f"Slope (m): {slope:.4f}")
    print(f"Intercept (c): {intercept:.4f}")
    
    metrics = evaluate_regression_model(model, X_dummy, y_dummy)
    print("Evaluation:", metrics)
