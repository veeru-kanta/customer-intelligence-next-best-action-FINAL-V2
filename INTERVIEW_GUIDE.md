# Interview Guide

## 1. What problem did you solve?
The goal was to move from descriptive customer reporting to a decision system that predicts churn risk, mines customer feedback, and recommends prioritized interventions.

## 2. Why ROC-AUC?
Churn is a binary classification problem and the positive class can be uneven. ROC-AUC evaluates ranking quality across classification thresholds rather than depending on one arbitrary threshold.

## 3. Why statistical testing?
Prediction alone does not establish whether an observed relationship is statistically meaningful. The project uses hypothesis tests to validate selected behavioral relationships.

## 4. Why NLP?
Customer feedback is unstructured. TF-IDF and clustering provide a simple, interpretable way to surface recurring themes without requiring a large labeled dataset.

## 5. Why next-best-action?
A model probability is not a business decision. The decision layer converts risk into a prioritized action list under a limited campaign capacity.

## 6. How would you improve this in production?
I would add probability calibration, temporal validation, customer-level leakage checks, uplift modeling/A-B tests, a formal optimization solver, model monitoring, data drift checks, and MLflow Model Registry.
