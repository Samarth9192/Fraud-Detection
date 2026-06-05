# Fraud-Detection

An end-to-end machine learning system to detect fraudulent credit card transactions with real-time prediction, SHAP explainability, and fraud pattern analysis.

**Live Demo:** https://fraud-detection-py.streamlit.app/

---

## Problem

Credit card fraud is rare but costly. Only 0.17% of transactions are fraudulent, making it a classic imbalanced classification problem. The extreme class imbalance makes accuracy a misleading metric, requiring evaluation through precision, recall, F1-score, and ROC-AUC.

---

## Dataset

- **Source:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- 284,807 transactions over 48 hours
- 492 fraud cases (0.17%)
- Features V1–V28 are PCA-transformed for privacy. Time and Amount are the only interpretable features.

---

## What This System Does

- Detects fraud using a trained Random Forest classifier
- Handles severe class imbalance using SMOTE oversampling
- Explains individual predictions using SHAP waterfall plots
- Clusters fraud transactions into behavioural patterns using KMeans
- Deployed as an interactive Streamlit dashboard with two modes:

  - **Random Simulator** — pick real transactions from the dataset and watch the model predict
  - **Manual Prediction** — enter amount, time of day, and transaction category for instant prediction

---

## Model Results

| Model | F1 (Fraud) | Precision | Recall | ROC-AUC |
|---------|---------|---------|---------|---------|
| **Random Forest** | **0.83** | **0.91** | **0.76** | **0.95** |
| XGBoost | 0.64 | 0.53 | 0.80 | 0.97 |
| Logistic Regression | 0.10 | 0.05 | 0.87 | 0.96 |
| Decision Tree | 0.06 | 0.03 | 0.85 | 0.94 |

Random Forest was selected as the final model. It achieves 91% precision, meaning that when the model flags a transaction as fraud, it is correct 91% of the time. XGBoost achieved a higher ROC-AUC but lower F1-score at the evaluated threshold.

---

## Technical Approach

### Class Imbalance — SMOTE

Training data contained 226,602 normal transactions and only 378 fraud transactions. SMOTE was applied only to the training data to balance classes while preserving the real-world fraud distribution in the test set.

### Threshold Tuning

Evaluated the default threshold of 0.5 using the precision-recall tradeoff. Since missed frauds are often more costly than false alarms, thresholds were explored to improve recall while maintaining reasonable precision.

### SHAP Explainability

Used SHAP TreeExplainer to explain individual predictions. For any flagged transaction, a waterfall plot shows which features contributed most toward or away from a fraud prediction.

### Fraud Pattern Clustering

Applied KMeans (k=3) on fraud transactions to identify behavioural patterns.

| Cluster | Count | Mean Amount | Time Pattern |
|---------|---------|---------|---------|
| Cluster 2 | 191 | ₹95 | Early period |
| Cluster 0 | 147 | ₹137 | Mid period |
| Cluster 1 | 135 | ₹150 | Late period |

The clustering analysis revealed distinct groups of fraudulent transactions with differing transaction amounts and temporal patterns.

---

## Project Structure

```text
Fraud-Detection/
├── streamlit_app.py
├── rf_model.pkl
├── scaler.pkl
├── v_means.json
├── fraud_samples.csv
├── normal_samples.csv
├── requirements.txt
└── README.md
