[README.md](https://github.com/user-attachments/files/28266174/README.md)
# 🤖 Coordinated Campaign & Spam Bot Classifier

An industry-style, educational end-to-end Machine Learning pipeline and interactive dashboard designed to identify coordinated spam bot campaigns on social media. Built specifically to cover the **CSM354: Machine Learning-I** syllabus.

---

## 📈 Project Overview
This repository contains a hybrid Machine Learning prototype that analyzes social media comments. By merging **NLP feature spaces (TF-IDF, PCA)** with **account metadata (age, frequency, similarity)**, this tool builds and evaluates all core models from the CSM354 syllabus.

### 🎯 Key Syllabus Coverage
1. **Simple Linear Regression**: Predicts account engagement rates from posting frequency.
2. **Multiple Linear Regression**: Predicts account engagement using multiple numerical metadata inputs.
3. **Logistic Regression**: Classifies comments as Human vs. Bot (utilizes Sigmoid and MLE).
4. **Naive Bayes Text Classification**: Classifies text strings on TF-IDF vectors using Laplace smoothing.
5. **K-Means Clustering**: Groups accounts by minimizing Within-Cluster Sum of Squares (WCSS).
6. **Hierarchical Agglomerative Clustering**: Groups comments bottom-up and produces dendrogram links.
7. **DBSCAN Clustering**: Clusters accounts based on local density and filters outliers as Noise (-1).
8. **Principal Component Analysis (PCA)**: Performs dimensionality reduction to visualize high-dimensional word vectors in 2D.
9. **AutoML**: Automates model comparison and hyperparameter tuning.
10. **Model Suggestion using GenAI**: A rule-based recommendation advisor.

---

## 📁 Repository Structure
```text
csm354_ml_project/
│
├── data/
│   ├── raw/
│   │   └── social_comments.csv         # Raw simulated comments dataset (1,200 records)
│   └── processed/
│       └── clean_comments.csv          # Preprocessed data with PCA components
│
├── src/
│   ├── data_preprocessing.py           # Text cleaning, normalisation, and MinMax/Z-score scaling
│   ├── feature_engineering.py          # TF-IDF calculation, PCA reduction, and RFE selection
│   ├── utils.py                        # Hopkins statistic, L1/L2 distances, confusion metrics from scratch
│   ├── train_pipeline.py               # Complete model training script
│   └── models/
│       ├── simple_linear_regression.py # Slope & intercept fitting
│       ├── multiple_linear_regression.py # Multi-predictor coefficients fitting
│       ├── logistic_regression.py      # Probability classification & decision boundaries
│       ├── naive_bayes.py              # multinomial NB text classification
│       ├── kmeans_clustering.py        # K-Means centroid clustering
│       ├── hierarchical_clustering.py  # Dendrogram linkages
│       ├── dbscan_clustering.py        # Density core/border grouping
│       ├── automl.py                   # GridSearchCV hyperparameter comparison
│       └── genai_advisor.py            # Simulated GenAI advisory reports
│
├── app/
│   └── dashboard.py                    # Streamlit Dashboard code (Dark Mode, interactive graphs)
│
├── practicals/
│   ├── practicals_solver.py            # Console solver for the 12 syllabus practicals
│   └── practicals_notes.md             # Theoretical math formulations
│
├── docs/
│   ├── project_report.md               # High-scoring Project Report
│   ├── viva_prep.md                    # 50+ Q&A study guide
│   └── slides_content.md               # PPT slides outline
│
├── requirements.txt                    # Python packages list
└── README.md                           # This README file
```

---

## ⚙️ Installation & Local Setup

### 1. Prerequisites
Ensure you have **Python 3.8+** installed on your system.

### 2. Clone the Repository & Initialize Environment
Open your terminal in the project directory and run:
```bash
# Set up virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Generate Data & Train Models
Execute the pipeline to generate synthetic data, scale features, and train all models:
```bash
# Generate the dataset
python src/generate_dataset.py

# Run the complete model training pipeline
python src/train_pipeline.py
```
*This will create the `models_saved/` directory containing all trained binary files (`.joblib`).*

### 4. Run the Streamlit Dashboard
Launch the interactive dashboard locally:
```bash
streamlit run app/dashboard.py
```
*A browser window should automatically open at `http://localhost:8501`.*

### 5. Run the Practical Exam Solvers
To view the step-by-step mathematical answers for the 12 syllabus practicals:
```bash
python practicals/practicals_solver.py
```

---

## 🏆 Performance Overview
- **Regression models**: Explains ~88.4% of variance ($R^2$ score).
- **Metadata Classifier (Logistic Regression)**: Achieves ~100% accuracy.
- **Text Classifier (Naive Bayes)**: Achieves ~78.7% accuracy on raw comment strings.
- **Clustering**: High clustering tendency verified by a **Hopkins Statistic** of **0.9471**, with a **Silhouette Score** of **0.7376** for K-Means.
