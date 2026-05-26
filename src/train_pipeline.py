"""
Filename: src/train_pipeline.py
Purpose: Run the complete ML pipeline from dataset loading to model training, evaluation, and saving.
Explanation:
- Loads raw dataset data/raw/social_comments.csv.
- Preprocesses data (cleans text, scales numeric attributes).
- Extracts features (TF-IDF vectorizer, PCA projection).
- Performs Feature Selection: Correlation study & RFE.
- Trains ALL models:
  - Simple Linear Regression (Frequency -> Engagement)
  - Multiple Linear Regression (Metadata -> Engagement)
  - Logistic Regression (Metadata -> is_it_cell)
  - Naive Bayes (TF-IDF -> is_it_cell)
  - K-Means & Hierarchical & DBSCAN (Clustering)
- Runs AutoML (Grid Search) and prints suggestions.
- Saves model artifacts to models_saved/ for dashboard inference.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import local modules
from data_preprocessing import preprocess_dataframe
from feature_engineering import compute_tfidf, apply_pca, perform_correlation_selection, perform_rfe_selection
from utils import calculate_confusion_matrix_metrics, calculate_hopkins_statistic

# Import models
from models.simple_linear_regression import train_simple_linear_regression, evaluate_regression_model
from models.multiple_linear_regression import train_multiple_linear_regression, evaluate_multiple_regression
from models.logistic_regression import train_logistic_regression, evaluate_logistic_model
from models.naive_bayes import train_naive_bayes, evaluate_naive_bayes
from models.kmeans_clustering import fit_kmeans
from models.hierarchical_clustering import fit_hierarchical_clustering
from models.dbscan_clustering import fit_dbscan
from models.automl import run_automl
from models.genai_advisor import generate_genai_suggestions

def run_pipeline():
    # 1. Load Dataset
    raw_path = "data/raw/social_comments.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}. Run generate_dataset.py first.")
        
    print("[PIPELINE] Loading raw comments dataset...")
    df = pd.read_csv(raw_path)
    print(f" -> Loaded {len(df)} records.")
    
    # 2. Data Cleaning & Preprocessing
    print("\n[PIPELINE] Running text cleaning and numeric feature scaling...")
    df_processed, age_scaler, freq_scaler = preprocess_dataframe(df)
    
    # Save processed dataframe for Streamlit
    os.makedirs("data/processed", exist_ok=True)
    df_processed.to_csv("data/processed/clean_comments.csv", index=False)
    print(" -> Saved clean_comments.csv to data/processed/")
    
    # 3. Feature Extraction (TF-IDF & PCA)
    print("\n[PIPELINE] Computing TF-IDF word vectors for comment texts...")
    tfidf_matrix, tfidf_vectorizer, vocab = compute_tfidf(df_processed['cleaned_text'], max_features=50)
    print(f" -> TF-IDF matrix shape: {tfidf_matrix.shape}. Vocabulary size: {len(vocab)}")
    
    print("[PIPELINE] Applying Principal Component Analysis (PCA)...")
    pca_features, pca_model = apply_pca(tfidf_matrix, n_components=2)
    df_processed['pca_1'] = pca_features[:, 0]
    df_processed['pca_2'] = pca_features[:, 1]
    
    # 4. Feature Selection
    print("\n[PIPELINE] Feature Selection...")
    numeric_features = ['scaled_age', 'scaled_frequency', 'scaled_similarity']
    target_class = 'is_it_cell'
    
    # Correlation Selection
    corr_scores = perform_correlation_selection(df_processed, numeric_features, target_class)
    
    # Recursive Feature Elimination (RFE)
    # Target label: is_it_cell, Inputs: scaled numeric columns
    X_num = df_processed[numeric_features]
    y_class = df_processed[target_class]
    
    selected_mask, ranking = perform_rfe_selection(X_num, y_class, n_features_to_select=2)
    print("\nRFE Feature Rankings (1 means selected):")
    for col, rank in zip(numeric_features, ranking):
        print(f"- {col}: Rank {rank}")
        
    # 5. Train-Test Split (Classification & Regression)
    # We split into 80% train and 20% test
    X_train_num, X_test_num, y_train_class, y_test_class = train_test_split(
        X_num, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    X_train_text, X_test_text, _, _ = train_test_split(
        tfidf_matrix, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # Target regression values (engagement rate)
    y_reg = df_processed['engagement_rate']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_num, y_reg, test_size=0.2, random_state=42
    )
    
    # 6. Model Training & Evaluation
    print("\n" + "="*50)
    print(" TRAINING ALL MODELS ")
    print("="*50)
    
    # (a) Simple Linear Regression
    print("\n--- Model 1: Simple Linear Regression ---")
    # Using scaled_frequency to predict engagement_rate
    X_train_simple = X_train_reg[['scaled_frequency']]
    X_test_simple = X_test_reg[['scaled_frequency']]
    
    simple_reg, slope, intercept = train_simple_linear_regression(X_train_simple, y_train_reg)
    simple_eval = evaluate_regression_model(simple_reg, X_test_simple, y_test_reg)
    print(f"Equation: Engagement = {slope:.4f} * ScaledFrequency + {intercept:.4f}")
    print("Evaluation:", simple_eval)
    
    # (b) Multiple Linear Regression
    print("\n--- Model 2: Multiple Linear Regression ---")
    multi_reg, coeffs, multi_intercept = train_multiple_linear_regression(X_train_reg, y_train_reg)
    multi_eval = evaluate_multiple_regression(multi_reg, X_test_reg, y_test_reg)
    print("Coefficients:", dict(zip(numeric_features, coeffs)))
    print("Evaluation:", multi_eval)
    
    # (c) Logistic Regression
    print("\n--- Model 3: Logistic Regression ---")
    logistic_clf, log_weights, log_intercept = train_logistic_regression(X_train_num, y_train_class)
    log_eval = evaluate_logistic_model(logistic_clf, X_test_num, y_test_class)
    log_metrics = calculate_confusion_matrix_metrics(y_test_class, log_eval["predictions"])
    print("Weights:", dict(zip(numeric_features, log_weights)))
    print("Evaluation Confusion Matrix:", log_metrics)
    
    # (d) Naive Bayes Text Classification
    print("\n--- Model 4: Naive Bayes Text Classification ---")
    nb_clf = train_naive_bayes(X_train_text, y_train_class, alpha=1.0)
    nb_eval = evaluate_naive_bayes(nb_clf, X_test_text, y_test_class)
    nb_metrics = calculate_confusion_matrix_metrics(y_test_class, nb_eval["predictions"])
    print("Evaluation Confusion Matrix:", nb_metrics)
    
    # (e) K-Means Clustering (Unsupervised - using scaled numerical columns)
    print("\n--- Model 5: K-Means Clustering ---")
    hopkins_val = calculate_hopkins_statistic(X_num)
    print(f"Hopkins Statistic (Clustering tendency): {hopkins_val:.4f}")
    
    kmeans_model, km_labels, km_centroids, km_wcss, km_sil = fit_kmeans(X_num, n_clusters=2)
    print(f"WCSS (Inertia): {km_wcss:.4f}")
    print(f"Silhouette Score: {km_sil:.4f}")
    
    # (f) Hierarchical Clustering
    print("\n--- Model 6: Hierarchical Clustering ---")
    hc_model, hc_labels, hc_sil = fit_hierarchical_clustering(X_num, n_clusters=2, linkage_type='ward')
    print(f"Silhouette Score (Hierarchical): {hc_sil:.4f}")
    
    # (g) DBSCAN Clustering
    print("\n--- Model 7: DBSCAN Clustering ---")
    db_model, db_labels, db_clusters, db_noise, db_sil = fit_dbscan(X_num, eps=0.3, min_samples=10)
    print(f"Found {db_clusters} clusters and {db_noise} noise points.")
    print(f"Silhouette Score (DBSCAN): {db_sil:.4f}")
    
    # (h) AutoML (Grid Search Classification Selection)
    print("\n--- AutoML Model Tuning ---")
    best_clf, best_name, automl_results = run_automl(X_train_num, y_train_class)
    
    # (i) GenAI Advisor Guidance
    print("\n--- GenAI Model Suggestion Advisor ---")
    # Linearity is highest correlation with regression target
    max_linearity = df_processed[numeric_features].corrwith(df_processed['engagement_rate']).abs().max()
    meta = {
        'num_samples': len(df_processed),
        'num_features': len(numeric_features) + tfidf_matrix.shape[1],
        'has_text': True,
        'task_type': 'classification',
        'linearity_score': max_linearity,
        'hopkins_statistic': hopkins_val,
        'dimensions': X_num.shape[1]
    }
    advice = generate_genai_suggestions(meta)
    print(advice)
    
    # 7. Save Model Binaries
    print("\n[PIPELINE] Saving models to models_saved/ ...")
    os.makedirs("models_saved", exist_ok=True)
    
    # Save standard classification models & preprocess steps
    joblib.dump(logistic_clf, "models_saved/logistic_regression_model.joblib")
    joblib.dump(nb_clf, "models_saved/naive_bayes_model.joblib")
    joblib.dump(kmeans_model, "models_saved/kmeans_model.joblib")
    joblib.dump(tfidf_vectorizer, "models_saved/tfidf_vectorizer.joblib")
    joblib.dump(pca_model, "models_saved/pca_model.joblib")
    joblib.dump(age_scaler, "models_saved/age_scaler.joblib")
    joblib.dump(freq_scaler, "models_saved/freq_scaler.joblib")
    joblib.dump(simple_reg, "models_saved/simple_regression_model.joblib")
    joblib.dump(multi_reg, "models_saved/multiple_regression_model.joblib")
    
    print("[SUCCESS] All models saved successfully! Pipeline complete.")

if __name__ == "__main__":
    run_pipeline()
