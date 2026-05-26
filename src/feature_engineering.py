"""
Filename: src/feature_engineering.py
Purpose: Transform preprocessed data into features for ML modeling.
Explanation:
- Computes TF-IDF (Term Frequency-Inverse Document Frequency) vectors for text comments.
- Implements PCA (Principal Component Analysis) to reduce TF-IDF dimensionality.
- Performs Feature Selection using:
  1. Pearson Correlation
  2. RFE (Recursive Feature Elimination)
  3. PCA Component Importance
- Written in beginner-friendly, clean python code with detailed comments.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def compute_tfidf(corpus, max_features=100):
    """
    Computes TF-IDF features from text corpus.
    TF-IDF scores show how important a word is to a document relative to the whole corpus.
    Formula: TF(t,d) * IDF(t)
    - TF (Term Frequency): count of term t in document d / total words in d
    - IDF (Inverse Document Frequency): log(Total documents / Documents containing term t)
    """
    # Initialize the TF-IDF vectorizer. We limit max features to 100 for simplicity and clustering performance.
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    
    # Fit and transform the corpus (learn vocab, return document-term matrix)
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    
    # Convert sparse matrix to dense array
    tfidf_dense = tfidf_matrix.toarray()
    
    # Get vocabulary word names
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    return tfidf_dense, tfidf_vectorizer, feature_names

from models.pca_reduction import fit_pca

def apply_pca(features_matrix, n_components=2):
    """
    Applies Principal Component Analysis (PCA) to reduce dimensionality.
    Uses the dedicated PCA module matching the CSM354 syllabus.
    """
    pca_model, reduced_features, _ = fit_pca(features_matrix, n_components=n_components)
    return reduced_features, pca_model

def perform_correlation_selection(df, numeric_cols, target_col):
    """
    Identifies features highly correlated with target column.
    Uses Pearson correlation coefficient. Values range from -1 to 1.
    """
    correlation_matrix = df[numeric_cols + [target_col]].corr()
    target_correlations = correlation_matrix[target_col].drop(target_col)
    
    print("\nFeature Correlation with Target Variable:")
    for feature, val in target_correlations.items():
        print(f"- {feature}: {val:.4f}")
        
    return target_correlations

def perform_rfe_selection(X, y, n_features_to_select=2):
    """
    Performs Recursive Feature Elimination (RFE) to select top numerical features.
    RFE fits a model (e.g., Logistic Regression) and recursively removes the least important features 
    based on feature weights (coefficients) until the target number of features is reached.
    """
    estimator = LogisticRegression()
    # Initialize RFE selector
    rfe_selector = RFE(estimator, n_features_to_select=n_features_to_select)
    rfe_selector.fit(X, y)
    
    # Support tells us if feature is selected (True/False)
    selected_mask = rfe_selector.support_
    # Ranking gives order of elimination (1 is best/selected)
    ranking = rfe_selector.ranking_
    
    return selected_mask, ranking

if __name__ == "__main__":
    print("Testing Feature Engineering modules...")
    corpus = [
        "support this leader now join today",
        "i love this new clean UI update",
        "support this brand clean government"
    ]
    matrix, vec, names = compute_tfidf(corpus, max_features=5)
    print("Vocabulary words:", names)
    print("TF-IDF matrix shape:", matrix.shape)
    
    reduced, pca = apply_pca(matrix, n_components=2)
    print("PCA reduced shape:", reduced.shape)
