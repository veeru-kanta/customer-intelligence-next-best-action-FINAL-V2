# Customer Intelligence & Next-Best-Action Engine

**End-to-end Data Science portfolio project for customer analytics and decision optimization.**

## Business problem
A business wants to understand customer behavior, identify customers at risk of churn, learn what customers are saying, and prioritize the most useful intervention when campaign resources are limited.

## What this project demonstrates
- Structured + unstructured customer data analysis
- SQL/Python-style data preparation
- Statistical hypothesis testing
- Customer segmentation / behavioral profiling
- Churn prediction with explainable modeling
- NLP and TF-IDF text mining
- MLflow experiment tracking
- Next-best-action prioritization
- Business-impact reporting

## Dataset
The repository uses a **synthetic, reproducible dataset of 25,000 customer observations**. It contains purchase behavior, engagement, support interactions, ratings, customer feedback, sentiment, product category and churn outcome.

No real customer or healthcare data is included.

## Architecture

`Synthetic data → Quality checks → Statistical analysis → ML/NLP → Customer risk → Next-best-action → Dashboard`

## Results
Run the pipeline locally to generate the measured model metrics and NLP clusters. **Resume metrics should be copied from the generated `reports/model_results.csv` rather than invented.**

## Quick start

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Outputs:
- `reports/model_results.csv`
- `reports/nlp_clusters.csv`
- `data/processed/next_best_actions.csv`
- `models/churn_pipeline.joblib`
- `mlruns/` for MLflow local experiment tracking when MLflow is installed
- `reports/measured_results.txt` for the measured model metrics

## Suggested Power BI pages
1. Executive overview: customers, revenue, churn rate, at-risk customers
2. Customer segments: behavior and value by segment
3. Churn drivers: risk distribution and important behavioral factors
4. Voice of customer: sentiment and feedback themes
5. Next-best-action: priority customers, recommended action and expected impact

## Interview story
The key design choice is to move beyond prediction. The model identifies risk, NLP explains customer concerns, statistical tests validate relationships, and the decision layer converts predictions into prioritized actions.

## Responsible-use note
This is a portfolio project using synthetic data. Recommendations are analytical demonstrations, not real customer decisions.
