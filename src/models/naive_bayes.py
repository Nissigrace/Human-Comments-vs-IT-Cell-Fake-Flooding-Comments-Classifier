"""
Filename: src/models/naive_bayes.py
Purpose: Implement Naive Bayes Text Classification for analyzing comment texts.
Syllabus connection: Unit 4 - Naive Bayes Text Classification.
Explanation:
- Naive Bayes is a probabilistic classifier based on Bayes' Theorem:
  P(Class | Document) = [ P(Document | Class) * P(Class) ] / P(Document)
- It assumes "naive" conditional independence: that features (words) do not affect each other given the class label.
- Multinomial Naive Bayes is used for discrete counts or TF-IDF values.
- Laplace Smoothing (Add-one smoothing): 
  Prevents zero probabilities for words that do not appear in the training dataset for a particular class.
  Formula: P(word_i | Class) = (count(word_i, Class) + alpha) / (total_count(Class) + alpha * vocab_size)
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def train_naive_bayes(X_train, y_train, alpha=1.0):
    """
    Fits a Multinomial Naive Bayes classifier on TF-IDF vectors.
    alpha represents the Laplace smoothing parameter (alpha=1.0 is standard).
    """
    # 1. Initialize the Multinomial Naive Bayes classifier
    model = MultinomialNB(alpha=alpha)
    
    # 2. Fit model to TF-IDF inputs and corresponding labels
    model.fit(X_train, y_train)
    
    return model

def evaluate_naive_bayes(model, X_test, y_test):
    """
    Evaluates Naive Bayes text classifier.
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=["Human", "IT Cell"], output_dict=True)
    
    return {
        "Accuracy": round(acc, 4),
        "Report": report,
        "predictions": predictions,
        "probabilities": probabilities
    }

if __name__ == "__main__":
    print("Testing Naive Bayes Classifier...")
    # Mock TF-IDF features for 4 documents and 3 vocabulary words
    # Vocab: ['leader', 'love', 'government']
    X_dummy = np.array([
        [0.9, 0.0, 0.8], # spam
        [0.8, 0.1, 0.9], # spam
        [0.0, 0.9, 0.1], # human
        [0.1, 0.8, 0.0]  # human
    ])
    y_dummy = np.array([1, 1, 0, 0])
    
    model = train_naive_bayes(X_dummy, y_dummy, alpha=1.0)
    print("Model log-prior probability:", model.class_log_prior_)
    print("Model log-likelihood features:", model.feature_log_prob_)
    
    results = evaluate_naive_bayes(model, X_dummy, y_dummy)
    print("Accuracy:", results["Accuracy"])
