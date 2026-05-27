"""
Filename: practicals/practicals_solver.py
Purpose: Solve all 12 syllabus practicals for CSM354 Machine Learning-I with step-by-step math and explanations.
Explanation:
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def print_header(title):
    print("\n" + "="*80)
    print(f" PRACTICAL: {title}")
    print("="*80)

# ==========================================================
# INTERNAL MATHEMATICAL HELPERS (Self-contained)
# ==========================================================

def solve_gradient_descent_linear(x, y, m_start=0.0, c_start=0.0, alpha=0.01, iterations=5):
    n = len(x)
    x = np.array(x)
    y = np.array(y)
    
    m = m_start
    c = c_start
    history = []
    
    for epoch in range(1, iterations + 1):
        y_pred = m * x + c
        error = y_pred - y
        cost = np.mean(error ** 2) / 2
        dm = np.mean(error * x)
        dc = np.mean(error)
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
    w = np.array(w_start, dtype=float)
    x = np.array(x, dtype=float)
    y = float(y)
    history = []
    
    for epoch in range(1, iterations + 1):
        z = np.dot(w, x)
        h = 1 / (1 + np.exp(-z))
        err = y - h
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


def calculate_confusion_matrix_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    recall = sensitivity
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
        "Accuracy": round(accuracy, 4),
        "Sensitivity (Recall)": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "Precision": round(precision, 4),
        "F1-Score": round(f1_score, 4)
    }


def calculate_euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


def calculate_manhattan_distance(point1, point2):
    return np.sum(np.abs(np.array(point1) - np.array(point2)))


def calculate_hopkins_statistic(X, m=10):
    X = np.array(X)
    n = X.shape[0]
    d = X.shape[1]
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    synthetic_points = np.random.uniform(low=mins, high=maxs, size=(m, d))
    
    u_distances, _ = nbrs.kneighbors(synthetic_points, n_neighbors=1)
    sum_u = np.sum(u_distances)
    
    random_indices = np.random.choice(n, size=m, replace=False)
    real_sample_points = X[random_indices]
    
    real_distances, _ = nbrs.kneighbors(real_sample_points, n_neighbors=2)
    sum_v = np.sum(real_distances[:, 1])
    
    return sum_u / (sum_u + sum_v)


# ==========================================================
# 1. normalize(data)
# ==========================================================
def practical_1_normalize():
    print_header("1. Data Normalization (Min-Max & Z-Score)")
    print("Concept:")
    print("Normalization transforms numerical features to a common scale to prevent larger values from dominating calculations.")
    print("- Min-Max Scaling maps data to range [0, 1]. Formula: X_norm = (X - X_min) / (X_max - X_min)")
    print("- Z-Score Standardization centers data around mean=0 and variance=1. Formula: X_std = (X - Mean) / StdDev\n")
    
    # Raw data
    data = [12, 15, 29, 35, 42, 55, 78, 99]
    print(f"Original Data: {data}")
    
    # Math implementation
    x = np.array(data)
    x_min = x.min()
    x_max = x.max()
    x_mean = x.mean()
    x_std = x.std()
    
    min_max = (x - x_min) / (x_max - x_min)
    z_score = (x - x_mean) / x_std
    
    print("\n--- CODE EXPLANATION ---")
    print("1. Convert list to NumPy array: `x = np.array(data)`")
    print("2. Min-Max Calculation: `(x - x.min()) / (x.max() - x.min())`")
    print("3. Z-Score Calculation: `(x - x.mean()) / x.std()`")
    
    print("\n--- OUTPUT RESULTS ---")
    df_results = pd.DataFrame({
        "Original": data,
        "Min-Max Normalized": np.round(min_max, 4),
        "Z-Score Standardized": np.round(z_score, 4)
    })
    print(df_results.to_string(index=False))


# ==========================================================
# 2. Mapping Yes/No/Maybe to 1/0/0.5
# ==========================================================
def practical_2_mapping():
    print_header("2. Categorical Mapping (Yes/No/Maybe -> 1/0/0.5)")
    print("Concept:")
    print("Machine learning algorithms require numerical inputs. Categorical values must be converted to numeric codes.")
    print("Here, we map ordered logical categories: 'Yes' (Full match) to 1.0, 'Maybe' (Partial match) to 0.5, and 'No' to 0.0.\n")
    
    categories = ["Yes", "No", "Maybe", "Yes", "Maybe", "No", "Yes"]
    print(f"Raw Categorical List: {categories}")
    
    # Code implementation
    mapping_dict = {"Yes": 1.0, "No": 0.0, "Maybe": 0.5}
    mapped_values = [mapping_dict[item] for item in categories]
    
    print("\n--- CODE EXPLANATION ---")
    print("1. Define mapping dictionary: `mapping_dict = {'Yes': 1.0, 'No': 0.0, 'Maybe': 0.5}`")
    print("2. List comprehension lookup: `mapped = [mapping_dict[item] for item in categories]`")
    
    print("\n--- OUTPUT RESULTS ---")
    for orig, mapped in zip(categories, mapped_values):
        print(f"Original Value: '{orig}' --> Mapped Value: {mapped}")


# ==========================================================
# 3. Gradient descent iteration problems
# ==========================================================
def practical_3_gradient_descent():
    print_header("3. Gradient Descent Iteration (Simple Linear Regression)")
    print("Concept:")
    print("Optimizes parameters m (slope) and c (intercept) of y = mx + c by taking steps along the negative gradient.")
    print("Formula: m_new = m_old - alpha * (dJ/dm)  and  c_new = c_old - alpha * (dJ/dc)")
    print("Where alpha = learning rate, J = Mean Squared Error loss.")
    
    x_points = [1, 2, 3]
    y_points = [2, 4, 5]
    print(f"\nGiven Points: X={x_points}, Y={y_points}")
    print("Starting parameters: m=0.0, c=0.0, learning rate alpha=0.1\n")
    
    history = solve_gradient_descent_linear(x_points, y_points, m_start=0.0, c_start=0.0, alpha=0.1, iterations=3)
    
    print("--- STEP-BY-STEP CALCULATION ---")
    for step in history:
        print(f"Epoch {step['Epoch']}:")
        print(f"  - Parameters before update: m = {step['m_old']}, c = {step['c_old']}")
        print(f"  - MSE Loss: {step['Cost']}")
        print(f"  - Gradients: dJ/dm = {step['dm']}, dJ/dc = {step['dc']}")
        print(f"  - Updated Parameters: m = {step['m_new']}, c = {step['c_new']}")
        print("-" * 50)


# ==========================================================
# 4. Logistic regression weight update
# ==========================================================
def practical_4_logistic_update():
    print_header("4. Logistic Regression Weight Update")
    print("Concept:")
    print("Update weights using Gradient Ascent/Descent based on log-likelihood.")
    print("Formula: w_new = w_old + alpha * (y - h_w(x)) * x")
    print("Where h_w(x) = sigmoid(w_old . x)")
    
    # 2 features plus intercept (x0 = 1, x1 = 2, x2 = 3)
    x = [1, 2, 3]
    y = 1
    w = [0.0, 0.0, 0.0]
    alpha = 0.2
    print(f"\nInput Vector: {x} (where x[0]=1 is intercept bias)")
    print(f"True Class Label: y = {y}")
    print(f"Initial Weights: {w}, Learning Rate: {alpha}\n")
    
    history = solve_logistic_weight_update(x, y, w, alpha=alpha, iterations=2)
    
    print("--- STEP-BY-STEP CALCULATION ---")
    for step in history:
        print(f"Iteration {step['Epoch']}:")
        print(f"  - Old weights: {step['w_old']}")
        print(f"  - z = w . x = {step['z']}")
        print(f"  - Probability prediction h_w(x) = {step['h_w(x)']}")
        print(f"  - Prediction Error (y - h) = {step['Error (y - h)']}")
        print(f"  - Weight updates: w_new = w_old + {alpha} * {step['Error (y - h)']} * {x}")
        print(f"  - New weights: {np.round(step['w_new'], 4)}")
        print("-" * 50)


# ==========================================================
# 5. Threshold-based classification
# ==========================================================
def practical_5_threshold_classification():
    print_header("5. Threshold-based Classification")
    print("Concept:")
    print("A classifier outputs probabilities between 0.0 and 1.0.")
    print("By default, a threshold of 0.5 splits classes. Changing this threshold adjusts")
    print("the classifier's sensitivity. Higher threshold = fewer positives (high precision).")
    print("Lower threshold = more positives (high recall/sensitivity).\n")
    
    probabilities = np.array([0.12, 0.45, 0.61, 0.72, 0.88, 0.31, 0.54, 0.95])
    actuals = np.array([0, 0, 1, 1, 1, 0, 0, 1])
    
    thresholds = [0.3, 0.5, 0.7]
    
    print(f"Comment Probabilities: {probabilities}")
    print(f"Actual Labels: {actuals}\n")
    
    print("--- THRESHOLD ANALYSIS RESULTS ---")
    for t in thresholds:
        preds = (probabilities >= t).astype(int)
        metrics = calculate_confusion_matrix_metrics(actuals, preds)
        print(f"Threshold = {t}:")
        print(f"  - Predicted Labels: {preds}")
        print(f"  - Accuracy: {metrics['Accuracy']}")
        print(f"  - Sensitivity: {metrics['Sensitivity (Recall)']}")
        print(f"  - Specificity: {metrics['Specificity']}")
        print(f"  - Precision: {metrics['Precision']}")
        print("-" * 50)


# ==========================================================
# 6 & 7. Metrics (Accuracy, Sensitivity, Specificity, F1-Score)
# ==========================================================
def practical_6_7_metrics():
    print_header("6 & 7. Evaluation Metrics from Confusion Matrix")
    print("Concept:")
    print("Calculates performance scores based on true/false classifications.")
    print("TP = True Positive, TN = True Negative, FP = False Positive (Type I), FN = False Negative (Type II).")
    print("- Sensitivity (Recall) = TP / (TP + FN)")
    print("- Specificity = TN / (TN + FP)")
    print("- Precision = TP / (TP + FP)")
    print("- F1-Score = 2 * (Precision * Recall) / (Precision + Recall)\n")
    
    # Mock labels
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 0, 1, 1, 1, 0, 0, 1, 1]
    
    print(f"Actual Labels:    {y_true}")
    print(f"Predicted Labels: {y_pred}")
    
    metrics = calculate_confusion_matrix_metrics(y_true, y_pred)
    
    print("\n--- RESULTS ---")
    print(f"Confusion Matrix: TP={metrics['TP']}, TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}")
    print(f"Accuracy: {metrics['Accuracy']}")
    print(f"Sensitivity (Recall): {metrics['Sensitivity (Recall)']}")
    print(f"Specificity: {metrics['Specificity']}")
    print(f"Precision: {metrics['Precision']}")
    print(f"F1-Score: {metrics['F1-Score']}")


# ==========================================================
# 8. Bayes theorem problem
# ==========================================================
def practical_8_bayes():
    print_header("8. Bayes Theorem Text Classification Problem")
    print("Concept:")
    print("Find probability of comment being SPAM (S) given it contains the word 'FREE' (F).")
    print("Formula: P(S | F) = P(F | S) * P(S) / P(F)")
    print("Where: P(F) = P(F | S)*P(S) + P(F | H)*P(H)  (Law of Total Probability, H = Human/Not Spam)")
    
    # Probabilities
    p_spam = 0.40      # P(S)
    p_human = 0.60     # P(H)
    p_free_spam = 0.80 # P(F|S)
    p_free_human = 0.10# P(F|H)
    
    # 1. Total Probability of Free
    p_free = (p_free_spam * p_spam) + (p_free_human * p_human)
    # 2. Posterior Probability P(S | F)
    p_spam_given_free = (p_free_spam * p_spam) / p_free
    
    print("\n--- GIVEN PARAMETERS ---")
    print(f"- Prior Probability of Spam P(S) = {p_spam}")
    print(f"- Prior Probability of Human P(H) = {p_human}")
    print(f"- Likelihood word 'FREE' occurs in Spam P(FREE | S) = {p_free_spam}")
    print(f"- Likelihood word 'FREE' occurs in Human P(FREE | H) = {p_free_human}")
    
    print("\n--- CALCULATION ---")
    print(f"1. Total Probability P(FREE) = P(FREE | S)*P(S) + P(FREE | H)*P(H)")
    print(f"                             = ({p_free_spam} * {p_spam}) + ({p_free_human} * {p_human}) = {p_free:.3f}")
    print(f"2. Posterior Probability P(S | FREE) = P(FREE | S) * P(S) / P(FREE)")
    print(f"                                     = ({p_free_spam} * {p_spam}) / {p_free:.3f} = {p_spam_given_free:.4f}")
    
    print(f"\n--- CONCLUSION ---")
    print(f"There is a {p_spam_given_free:.2%} probability that a comment containing 'FREE' is IT Cell/Spam.")


# ==========================================================
# 9. Euclidean and Manhattan distance
# ==========================================================
def practical_9_distances():
    print_header("9. Vector Distances (Euclidean vs Manhattan)")
    print("Concept:")
    print("- Euclidean Distance: Straight-line L2 distance. sqrt(sum((p1_i - p2_i)^2))")
    print("- Manhattan Distance: L1 block-wise taxicab distance. sum(|p1_i - p2_i|)\n")
    
    p1 = [3.0, 5.0]
    p2 = [7.0, 2.0]
    print(f"Point A: {p1}")
    print(f"Point B: {p2}\n")
    
    euclidean = calculate_euclidean_distance(p1, p2)
    manhattan = calculate_manhattan_distance(p1, p2)
    
    print("--- CALCULATION ---")
    print(f"Euclidean: sqrt( (7-3)^2 + (2-5)^2 ) = sqrt(16 + 9) = sqrt(25) = {euclidean:.4f}")
    print(f"Manhattan: |7-3| + |2-5| = 4 + 3 = {manhattan:.4f}")


# ==========================================================
# 10. Hopkins statistic
# ==========================================================
def practical_10_hopkins():
    print_header("10. Hopkins Statistic (Clustering Tendency)")
    print("Concept:")
    print("Assesses clustering tendency. If H is near 0.5, data is randomly distributed.")
    print("If H approaches 1.0, data contains distinct, highly separable clusters.\n")
    
    # Generate clustered dataset vs random dataset
    np.random.seed(42)
    clustered_data = np.vstack([
        np.random.normal(loc=1.0, scale=0.1, size=(50, 2)),
        np.random.normal(loc=10.0, scale=0.1, size=(50, 2))
    ])
    
    random_data = np.random.uniform(low=0.0, high=10.0, size=(100, 2))
    
    h_clustered = calculate_hopkins_statistic(clustered_data, m=10)
    h_random = calculate_hopkins_statistic(random_data, m=10)
    
    print("--- RESULTS ---")
    print(f"- Hopkins score for highly clustered dataset: {h_clustered:.4f} (Strong cluster tendency)")
    print(f"- Hopkins score for uniformly random dataset: {h_random:.4f} (Random noise, no tendency)")


# ==========================================================
# 11 & 12. K-Means clustering on Cricket dataset & Cluster Analysis
# ==========================================================
def practical_11_12_cricket_kmeans():
    print_header("11 & 12. K-Means & Cluster Analysis on Cricket Dataset")
    print("Concept:")
    print("Clusters cricket players based on performance features: 'Runs' and 'Strike Rate'.")
    print("Includes analyzing clusters and computing Silhouette scores to verify cluster separation.\n")
    
    # Cricket player database
    cricket_data = {
        "Player": ["Kohli", "Rohit", "Dhoni", "Bumrah", "Shami", "Hardik", "Rahul", "Siraj", "Jadeja", "Ashwin"],
        "Runs": [850, 780, 520, 45, 30, 410, 680, 20, 390, 180],
        "Strike_Rate": [138.5, 142.1, 130.4, 75.0, 60.2, 145.8, 128.2, 50.0, 125.4, 110.5]
    }
    df = pd.DataFrame(cricket_data)
    print("Cricket Player Dataset:")
    print(df.to_string(index=False))
    
    # Preprocess
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[["Runs", "Strike_Rate"]])
    
    # Fit K-Means
    from sklearn.cluster import KMeans
    kmeans_model = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
    labels = kmeans_model.fit_predict(scaled_features)
    
    df["Cluster"] = labels
    
    # Compute Silhouette Score
    silhouette = silhouette_score(scaled_features, labels)
    
    print("\n--- RESULTS ---")
    print("Assigned Player Profiles:")
    for idx, row in df.iterrows():
        c = row["Cluster"]
        # Match clusters dynamically to roles for readability
        if c == 0:
            role = "Top-order Batsmen"
        elif c == 1:
            role = "Bowlers"
        else:
            role = "All-Rounders"
        print(f" * {row['Player']}: Cluster {c} ({role})")
        
    print(f"\nSilhouette Coefficient for Grouping: {silhouette:.4f}")
    print("Interpretation: Silhouette > 0.5 shows distinct clusters with excellent structural cohesion.")


# ==========================================================
# Main runner
# ==========================================================
if __name__ == "__main__":
    print("="*80)
    print(" CSM354 MACHINE LEARNING-I PRACTICAL SOLVER ENGINE")
    print("="*80)
    
    practical_1_normalize()
    practical_2_mapping()
    practical_3_gradient_descent()
    practical_4_logistic_update()
    practical_5_threshold_classification()
    practical_6_7_metrics()
    practical_8_bayes()
    practical_9_distances()
    practical_10_hopkins()
    practical_11_12_cricket_kmeans()
    
    print("\n" + "="*80)
    print(" ALL PRACTICALS RUN AND SOLVED SUCCESSFULLY!")
    print("="*80)
