# Telco Customer Churn Prediction

## Overview
This project builds an end-to-end machine learning classification pipeline to predict customer churn in a telecommunications company. By identifying customers who are likely to cancel their subscriptions, the business can implement proactive retention strategies, reduce recurring revenue loss, and optimize customer lifetime value.

## Project Structure
```text
├── customer_churn_prediction.ipynb   # Main Jupyter Notebook (EDA, Preprocessing, Modeling, Evaluation)
├── report.md                         # Comprehensive technical and business report
├── README.md                         # Project guide and documentation
└── Telco-Customer-Churn.csv          # Dataset (Telco Customer Churn)
```
## Prerequisites & Installation

Ensure you have Python installed along with the required scientific computing and machine learning libraries. You can install the necessary dependencies using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## How to Run the Project

1. Clone the repository or download the project files to your local machine.

2. Ensure the dataset (`Telco-Customer-Churn.csv`) is located in the root directory.

3. Launch Jupyter Notebook:
```bash
jupyter notebook customer_churn_prediction.ipynb
```

4. Run all cells sequentially from Task 1 to Task 6 to execute the complete workflow, including data loading, EDA, pipeline creation, model training, hyperparameter tuning, and threshold analysis.

## Methodology & Technical Highlights

- Data Leakage Prevention: Built modular `Pipelines` and `ColumnTransformer` using Scikit-Learn to ensure all preprocessing steps (imputation, scaling, and one-hot encoding) are fitted exclusively on the training set.

- Model Comparison: Evaluated multiple algorithms (Baseline, Logistic Regression, Decision Tree, Random Forest, and KNN) using comparable train/test splits.

- Metric Focus: Prioritized Recall for the positive class (`Churn = Yes`) to minimize false negatives (missed at-risk customers).

- Hyperparameter Tuning: Optimized the best-performing model using `GridSearchCV` with 5-fold cross-validation.

## Key Business Insights & Recommendations

- Contract Type: Month-to-month subscribers exhibit significantly higher churn rates; offering special incentives to transition them to annual contracts can drastically improve retention.

- Early Lifecycle: Churn is heavily concentrated among new customers (tenure under 12 months), highlighting the necessity of robust early-stage onboarding and support.

## Author

Developed as a comprehensive data science and machine learning predictive modeling project.

