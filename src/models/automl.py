"""
Filename: src/models/automl.py
Purpose: Implement an automated machine learning (AutoML) framework.
Syllabus connection: Unit 9 - AutoML.
Explanation:
- AutoML automates the ML pipeline tasks: preprocessing, algorithm selection, and hyperparameter tuning.
- In this script, we create a mini AutoML engine that:
  1. Accepts training features and targets.
  2. Iterates through multiple candidate models (Logistic Regression, Multinomial Naive Bayes, and extra model classes).
  3. Conducts hyperparameter search (GridSearchCV) for each model.
  4. Ranks the models based on cross-validation accuracy score.
  5. Automatically returns the best fitted estimator.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def run_automl(X_train, y_train):
    """
    Runs automated grid search across Logistic Regression and Gaussian Naive Bayes,
    returning the best candidate.
    """
    print("\n[AutoML] Initiating Model Search...")
    
    # 1. Define model pool and search grids
    # NOTE: We use GaussianNB here instead of MultinomialNB. 
    # Why? Because numeric features scaled with StandardScaler can contain negative values.
    # MultinomialNB is count-based and throws errors on negative values.
    # GaussianNB is designed for continuous variables and accepts negative inputs.
    model_pool = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["liblinear", "lbfgs"]
            }
        },
        "GaussianNaiveBayes": {
            "model": GaussianNB(),
            "params": {
                "var_smoothing": [1e-9, 1e-8, 1e-7]
            }
        }
    }
    
    results = {}
    best_score = -1
    best_model_name = ""
    best_estimator = None
    
    # 2. Iterate and fit
    for name, config in model_pool.items():
        print(f"[AutoML] Tuning hyper-parameters for: {name}")
        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=3,  # 3-Fold Cross-Validation
            scoring="accuracy",
            n_jobs=-1
        )
        
        # Fit Grid Search
        grid.fit(X_train, y_train)
        
        score = grid.best_score_
        best_params = grid.best_params_
        estimator = grid.best_estimator_
        
        results[name] = {
            "Best_CV_Score": round(score, 4),
            "Best_Params": best_params,
            "Estimator": estimator
        }
        
        print(f" -> Best CV Accuracy: {score:.4f} with params {best_params}")
        
        # 3. Save best estimator
        if score > best_score:
            best_score = score
            best_model_name = name
            best_estimator = estimator
            
    print(f"\n[AutoML SUCCESS] Champion Model Selected: {best_model_name} (CV Accuracy: {best_score:.4f})")
    
    return best_estimator, best_model_name, results

if __name__ == "__main__":
    print("Testing AutoML engine...")
    X_dummy = np.random.uniform(0, 1, size=(50, 4))
    # labels
    y_dummy = np.random.choice([0, 1], size=50)
    
    best_model, name, results = run_automl(X_dummy, y_dummy)
    print("Best overall model:", name)
