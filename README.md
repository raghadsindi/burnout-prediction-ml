# Occupational Burnout Prediction

Predictive modeling of occupational burnout levels in the tech industry using
supervised machine learning.

## Overview
Built a 3-class classifier (Low / Moderate / High burnout) on a 150,000-record
dataset with 24 features covering workload, lifestyle, and psychological health.
The direct burnout score was deliberately removed before training to prevent data
leakage, forcing the model to learn from genuine behavioral and psychological
signals instead.

## Models & Results
| Model | Accuracy | Weighted F1 |
|---|---|---|
| SVM (best) | 93.42% | 93.2% |
| Logistic Regression | 93.38% | 93.1% |
| Random Forest | 92.96% | 92.7% |
| Tuned Random Forest | ~93.0% | ~92.8% |

Stratified 70/30 train-test split used to preserve class distribution given
severe class imbalance (Low: 92,134 — Moderate: 12,820 — High: 46).

## Ethical Considerations
Mental-health prediction models carry real responsibility. This project's report
covers data privacy, algorithmic bias/fairness across demographic groups,
explainability (SHAP/LIME), and safeguards against misuse — model outputs are
framed strictly as support signals, never punitive or clinical decisions.

## Tools
Python, Pandas, Scikit-learn, Matplotlib, Seaborn, Google Colab

## Files
- `BurnOut_Project.ipynb` — full notebook: EDA, preprocessing, model training & evaluation
- `BurnOut_Project.pdf` — written report with confusion matrices and ethical analysis
