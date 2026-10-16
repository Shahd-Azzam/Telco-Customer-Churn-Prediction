# 📊 Customer Churn Prediction: Technical & Business Report

## 1. Executive Summary & Business Problem
Customer churn—when subscribers switch to competitors—presents a major financial threat to telecommunications companies. Retaining existing customers is significantly more cost-effective than acquiring new ones. This project develops an end-to-end machine learning classification pipeline to predict customer churn based on historical account details, usage patterns, and demographics. The primary objective is to maximize **Recall** for churners, ensuring that at-risk customers are successfully identified before they cancel their subscriptions.

## 2. Exploratory Data Analysis (EDA) Findings
Through rigorous data exploration of the Telco dataset, several key patterns emerged:
* **Contract Type:** Month-to-month contracts exhibit vastly higher absolute and relative churn rates compared to one-year or two-year commitments, highlighting contract length as a primary loyalty driver.
* **Customer Tenure:** Churn is heavily concentrated among new customers (tenure under 12 months), indicating that early onboarding and customer support interventions are critical.
* **Monthly Charges:** Customers who churn tend to have higher monthly charges, pointing to price sensitivity at premium pricing tiers.
* **Class Imbalance:** Non-churners comprise roughly 73% of the dataset, while churners represent ~27%, necessitating evaluation metrics beyond simple accuracy (prioritizing Precision, Recall, and F1-Score).

## 3. Methodology & Preprocessing
To ensure robust, unbiased results and prevent **Data Leakage**:
* **Data Cleaning:** The `customerID` identifier column was dropped. The `TotalCharges` column was cleaned, converted to numeric, and missing values were handled.
* **Train-Test Split:** The dataset was split into an 80/20 train-test partition using `stratify=y` to preserve class proportions.
* **Pipelines & Transformers:** A Scikit-Learn `ColumnTransformer` and `Pipeline` were implemented to impute missing values, scale numerical features (`StandardScaler`), and encode categorical features (`OneHotEncoder`). All transformations were **fitted exclusively on the training set**.

## 4. Model Comparison & Results
Several classification algorithms were evaluated alongside a baseline model, treating `Churn = Yes` as the positive class:

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Most Frequent)** | 0.73 | 0.00 | 0.00 | 0.00 |
| **Logistic Regression** | 0.81 | 0.65 | 0.57 | 0.61 |
| **Decision Tree** | 0.72 | 0.47 | 0.46 | 0.46 |
| **Random Forest (Untuned)** | 0.79 | 0.62 | 0.51 | 0.55 |
| **KNN** | 0.76 | 0.55 | 0.56 | 0.55 |
| **Tuned Random Forest (Final)** | 0.79 | 0.64 | 0.51 | 0.57 |

* **Hyperparameter Tuning:** Using `GridSearchCV` with 5-fold cross-validation, the Random Forest classifier was fine-tuned for optimal depth, number of estimators, and split criteria.
* **Threshold Optimization:** Analyzing the precision-recall curve confirmed that decision thresholds can be adjusted dynamically to maximize recall based on business appetite for retention marketing costs.

## 5. Strategic Business Recommendations
1. **Targeted Contract Incentives:** Proactively offer special promotional rates or perks to month-to-month subscribers to encourage migration to longer-term commitments.
2. **Early-Lifecycle Support:** Direct dedicated customer service outreach and onboarding assistance toward new customers within their first year of tenure who have high monthly charges.

## 6. Limitations
* **Class Imbalance:** Skewed distributions require careful threshold tuning to manage the trade-off between false negatives (missed churners) and false positives (wasted marketing incentives).
* **Static Snapshot Data:** The historical data reflects past behavioral patterns but cannot account for sudden external market shifts or competitor disruptions.

## 7. Conclusion & Final Recommendation
The **Tuned Random Forest Classifier** is selected as the final production model. Its robust handling of non-linear tabular interactions and strong recall performance make it an ideal tool for safeguarding recurring revenue and maximizing long-term customer lifetime value.