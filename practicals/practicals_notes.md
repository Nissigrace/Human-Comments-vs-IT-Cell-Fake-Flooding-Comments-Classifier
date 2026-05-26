# CSM354: Machine Learning-I Lab Record Notes

This document provides the theoretical formulas, conceptual answers, step-by-step mathematical calculations, and code explanations for all 12 syllabus practicals. Use this for your lab record and viva preparation.

---

## Practical 1: Data Normalization

### Concept
Machine Learning algorithms are sensitive to feature scales. Data Normalization adjusts values to a standard range.
1. **Min-Max Scaling**: Rescales data to the range $[0, 1]$.
2. **Z-Score Standardization**: Centers data around a mean of 0 and standard deviation of 1.

### Mathematical Formulas
1. **Min-Max**:
   $$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$
2. **Z-Score**:
   $$x_{std} = \frac{x - \mu}{\sigma}$$
   Where:
   - $\mu = \text{mean} = \frac{1}{N} \sum x_i$
   - $\sigma = \text{standard deviation} = \sqrt{\frac{1}{N} \sum (x_i - \mu)^2}$

---

## Practical 2: Categorical Mapping

### Concept
Text classifications require converting text tokens into numerical values. When categories are ordered (ordinal) or logical, mapping to floating points is preferred:
* "Yes" $\rightarrow 1.0$ (Strong Positive)
* "Maybe" $\rightarrow 0.5$ (Neutral/Partial)
* "No" $\rightarrow 0.0$ (Negative)

---

## Practical 3: Gradient Descent Iteration

### Concept
Gradient Descent is an iterative optimization algorithm used to minimize the cost function of Linear Regression.

### Mathematical Formulas
Given the hypothesis $y_{pred} = mx + c$ and Mean Squared Error cost:
$$J(m, c) = \frac{1}{2n} \sum_{i=1}^n (y_{pred}^{(i)} - y^{(i)})^2$$

The partial derivatives are:
$$\frac{\partial J}{\partial m} = \frac{1}{n} \sum (y_{pred} - y)x, \quad \frac{\partial J}{\partial c} = \frac{1}{n} \sum (y_{pred} - y)$$

Parameters update as:
$$m = m - \alpha \frac{\partial J}{\partial m}, \quad c = c - \alpha \frac{\partial J}{\partial c}$$

---

## Practical 4: Logistic Regression Weight Update

### Concept
Logistic regression parameters are learned by maximizing the likelihood function using gradient updates.

### Mathematical Formulas
The hypothesis is:
$$h_w(x) = g(w^T x) = \frac{1}{1 + e^{-w^T x}}$$

The weight update rule is:
$$w_j = w_j + \alpha (y - h_w(x)) x_j$$
Where:
- $\alpha$ is the learning rate.
- $(y - h_w(x))$ is the prediction error.

---

## Practical 5: Threshold-Based Classification

### Concept
Logistic regression outputs a probability $P(Y=1|X)$. The default classification threshold is $0.5$.
* Increasing the threshold (e.g., $0.7$) makes the model conservative, reducing False Positives (higher precision).
* Decreasing the threshold (e.g., $0.3$) makes the model aggressive, capturing more True Positives (higher recall/sensitivity) but increasing False Positives.

---

## Practical 6 & 7: Classification Evaluation Metrics

### Mathematical Formulas
Given a Confusion Matrix:
* **True Positive (TP)**: Predicted 1, Actual 1
* **True Negative (TN)**: Predicted 0, Actual 0
* **False Positive (FP)**: Predicted 1, Actual 0 (Type I Error)
* **False Negative (FN)**: Predicted 0, Actual 1 (Type II Error)

1. **Accuracy**:
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Sensitivity (Recall)**:
   $$\text{Sensitivity} = \frac{TP}{TP + FN}$$
3. **Specificity**:
   $$\text{Specificity} = \frac{TN}{TN + FP}$$
4. **Precision**:
   $$\text{Precision} = \frac{TP}{TP + FP}$$
5. **F1-Score** (Harmonic mean of precision and recall):
   $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## Practical 8: Bayes' Theorem

### Concept
Bayes' theorem calculates the probability of an event based on prior knowledge of conditions related to the event.

### Mathematical Formula
$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

Using the Law of Total Probability:
$$P(B) = P(B|A)P(A) + P(B|A')P(A')$$

---

## Practical 9: Vector Distance Metrics

### Mathematical Formulas
Given points $P = (p_1, p_2, \dots, p_n)$ and $Q = (q_1, q_2, \dots, q_n)$:
1. **Euclidean Distance (L2 Norm)**:
   $$d_{Euc} = \sqrt{\sum_{i=1}^n (p_i - q_i)^2}$$
2. **Manhattan Distance (L1 Norm)**:
   $$d_{Man} = \sum_{i=1}^n |p_i - q_i|$$

---

## Practical 10: Hopkins Statistic

### Concept
The Hopkins statistic determines if a dataset contains meaningful clusters.
* $H \approx 0.5$: The data is randomly distributed (no clustering tendency).
* $H \rightarrow 1.0$: The data is highly clustered (strong clustering tendency).

### Mathematical Formula
$$H = \frac{\sum_{i=1}^m u_i}{\sum_{i=1}^m u_i + \sum_{i=1}^m v_i}$$
Where:
- $u_i$ is the distance from a synthetic random point to its nearest real neighbor in the dataset.
- $v_i$ is the distance from a real random point in the dataset to its nearest neighbor.

---

## Practical 11 & 12: K-Means & Cluster Analysis

### Concept
K-Means partitions data into $K$ clusters. The algorithm:
1. Chooses $K$ random centroids.
2. Assigns points to the nearest centroid.
3. Recomputes centroids based on the mean of the assigned points.
4. Repeats until convergence.

The Silhouette Score is used to analyze cluster separation:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
Where:
- $a(i)$ is the mean intra-cluster distance of point $i$.
- $b(i)$ is the mean nearest-cluster distance for point $i$.
- $s(i) \in [-1, 1]$.
