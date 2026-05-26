"""
Filename: src/models/genai_advisor.py
Purpose: Implement Model Suggestion using GenAI concepts.
Syllabus connection: Unit 10 - Model Suggestion using GenAI.
Explanation:
- In production, large language models (LLMs) can act as AI advisors, taking in dataset descriptions, metadata, 
  and requirements to suggest the most appropriate ML architectures.
- This script implements a deterministic rule-and-heuristic engine simulating a GenAI Agent.
- It analyzes the dimensions, data types, correlation values, and clustering tendencies (Hopkins statistic) 
  of a dataset, and generates a structured, human-readable recommendation report matching the syllabus.
"""

def generate_genai_suggestions(dataset_meta):
    """
    Simulates a GenAI agent suggesting models based on dataset metadata.
    dataset_meta keys:
    - 'num_samples': int
    - 'num_features': int
    - 'has_text': bool
    - 'task_type': 'regression' or 'classification' or 'clustering'
    - 'linearity_score': float (correlation coefficient)
    - 'hopkins_statistic': float (clustering tendency)
    - 'dimensions': int (dimensionality count)
    """
    
    # 1. Start building the GenAI recommendation markdown response
    response = []
    response.append("### [GenAI] Model Suggestion Advisor")
    response.append("Based on the characteristics of your dataset, here is the architectural suggestion:\n")
    
    # 2. Add analysis of target task
    task = dataset_meta.get('task_type', 'classification')
    response.append(f"- **Detected Task Objective**: `{task.upper()}`")
    response.append(f"- **Dataset Shape**: {dataset_meta.get('num_samples')} samples, {dataset_meta.get('num_features')} features.")
    
    # 3. Decision tree suggestions simulating LLM prompt reasoning
    if task == 'classification':
        response.append("\n#### Recommended Classifiers:")
        if dataset_meta.get('has_text'):
            response.append(
                "1. **Naive Bayes Text Classification (Highly Recommended)**:\n"
                "   *Rationale*: Your dataset contains raw comment text. Text classifications are best handled by Naive Bayes "
                "because it processes sparse TF-IDF vectors efficiently using conditional independence assumptions and Laplace smoothing.\n"
            )
            response.append(
                "2. **Logistic Regression**:\n"
                "   *Rationale*: Great baseline model for binary classification (Human vs Fake). It utilizes Maximum Likelihood Estimation "
                "and produces probability scores for each comment, which lets us adjust classification decision thresholds dynamically.\n"
            )
        else:
            response.append(
                "1. **Logistic Regression (Recommended)**:\n"
                "   *Rationale*: Since there is no text corpus, we can directly fit a logistic model on numerical features like "
                "posting frequency and account age. It uses gradient descent weight updates to map inputs to a Sigmoid probability.\n"
            )
            
    elif task == 'regression':
        response.append("\n#### Recommended Regressors:")
        linearity = dataset_meta.get('linearity_score', 0.0)
        response.append(f"- **Linearity Analysis**: The feature correlation with target is {linearity:.2f}.")
        
        if abs(linearity) > 0.6:
            response.append(
                "1. **Simple Linear Regression**:\n"
                "   *Rationale*: There is a strong linear relationship. Using posting frequency as X and engagement rate as y "
                "will yield a highly accurate linear equation $y = mx + c$ with minimized Mean Squared Error.\n"
            )
        response.append(
            "2. **Multiple Linear Regression**:\n"
            "   *Rationale*: The engagement rate depends on multiple columns. Extending simple regression to a multi-variable linear model "
            "will allow fitting weights for multiple predictors simultaneously (e.g. account age + similarity + posting frequency).\n"
        )
        
    elif task == 'clustering':
        response.append("\n#### Recommended Clustering Models:")
        hopkins = dataset_meta.get('hopkins_statistic', 0.5)
        response.append(f"- **Clustering Tendency**: Hopkins Statistic is {hopkins:.4f}.")
        
        if hopkins > 0.7:
            response.append("   *Status*: The dataset has high clustering tendency (Hopkins > 0.7), meaning clusters exist naturally.\n")
        else:
            response.append("   *Status*: Warning! Hopkins statistic is close to 0.5, meaning data has random characteristics. Clustering may be noisy.\n")
            
        response.append(
            "1. **K-Means Clustering**:\n"
            "   *Rationale*: Efficient and simple. It partition-groups comments into $K$ clusters by minimizing Within-Cluster Sum of Squares (WCSS). "
            "Best evaluated using Silhouette Score to check cluster cohesiveness.\n"
        )
        response.append(
            "2. **DBSCAN (Density-Based Spatial Clustering)**:\n"
            "   *Rationale*: Since bot flooding activity occurs in highly dense groups while real comments are scattered, DBSCAN is ideal. "
            "It will group dense coordinates and automatically label anomalous individual accounts as Noise (-1) without requiring K to be specified.\n"
        )
        response.append(
            "3. **Hierarchical Agglomerative Clustering**:\n"
            "   *Rationale*: Helps inspect the grouping structure visually using a Dendrogram linkage chart. Useful for identifying bot-net sub-campaign structures.\n"
        )
        
    # 4. Dimensionality Reduction suggestion
    dims = dataset_meta.get('dimensions', 1)
    if dims > 5:
        response.append("\n#### Dimensionality Reduction:")
        response.append(
            "* **Principal Component Analysis (PCA)**:\n"
            f"  *Rationale*: Your dataset has high dimensions ({dims}). Applying PCA to compress features into 2 or 3 components "
            "will help visualize the clusters in 2D/3D charts and reduce noise for K-means/DBSCAN algorithms.\n"
        )
        
    return "\n".join(response)

if __name__ == "__main__":
    print("Testing GenAI Advisor...")
    meta = {
        'num_samples': 1200,
        'num_features': 105,
        'has_text': True,
        'task_type': 'clustering',
        'linearity_score': 0.12,
        'hopkins_statistic': 0.84,
        'dimensions': 100
    }
    
    advice = generate_genai_suggestions(meta)
    print(advice)
