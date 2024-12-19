This fraud prediction solution identifies potentially fraudulent transactions using machine learning models. 
The solution utilizes historical transaction data to build a binary classification model capable of distinguishing between fraudulent and non-fraudulent transactions. 

Key Features:
  - Structured and iterative approach to addressing imbalanced classification problems.
  - Emphasize on preprocessing flexibility, robust modeling, and comprehensive performance evaluation tailored to data characteristics.
  - Optimize and automate feature extraction, preprocessing, class balancer, and parameter/hyperparameters tuning and model selection, using a grid search with `roc_auc` as the scoring metric. 
  - Modular code structure for easy integration and updates.
  - Insights through visualizations.

Methodology
1. Data Exploration and visualizations.

2. Data Preprocessing:
  - Handle missing values and outliers.
  - Scale continuous variables and encode categorical features.
  - Transform skewed features.
  - High correlations features dropping.
  - Low-variance filtering.
  - Split data into training and testing sets ensuring no data leakage of clients and months. 

3. Feature Engineering
  - Create domain-specific features like hour or day of the week to capture temporal patterns.
  - Use techniques like label encoding and normalization.

4. Model Training
  - Compared RandomForest and XGBoost. RandomForest proved to be the best classifier for predicting fraud, providing the highest ROC-AUC
  - Handle imbalanced datasets using Upscaling/Downscaling.

5. Model Evaluation
  - Evaluate using ROC-AUC as the scoring metric, due to the imbalanced nature of the dataset. Scoring metric is a parameter.
  - Generate ROC-AUC curve and feature importance for visualization.

5. Results
  - ROC-AUC Score: 0.8275  
  - Demonstrates good performance, especially considering the data's imbalance.
