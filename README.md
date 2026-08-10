# Hospital Length of Stay Prediction (PRCP-1022-HospitalStayPred)

Predicting a patient's length of hospital stay using admission-time features — hospital, department, patient, and administrative data — to support proactive resource planning in a healthcare setting.

## Problem Statement
1. Prepare a complete data analysis report on hospital admission data.
2. Build a machine learning model to predict a patient's length of stay (bucketed into ranges from "0-10 days" to "More than 100 Days") using factors available at admission time.

## Project Type
Classification (11-class ordinal) — see [Challenges Report](./CHALLENGES_REPORT.md) for why this was framed as classification rather than regression, despite the problem statement's phrasing.

## Contents
- **`House Stay Prediction Solution.ipynb`** — Full analysis notebook: EDA, 15 visualizations (Univariate/Bivariate/Multivariate), hypothesis testing, feature engineering, and 3 ML models (Logistic Regression, Random Forest, XGBoost) with hyperparameter tuning.
- **`MODEL_COMPARISON_REPORT.md`** — Performance comparison across all three models and final model recommendation.
- **`CHALLENGES_REPORT.md`** — Data and modeling challenges encountered, techniques used, and reasoning.
- **`hospital_stay_prediction_model.joblib`** — Final trained Random Forest model, saved and sanity-checked for deployment.
- **`label_encoders.joblib`** — Saved label encoders needed to preprocess new/unseen data consistently.

## Dataset
~318,000 hospital admission records with 18 original features covering hospital attributes (type, region, department, ward), patient attributes (age, city, severity of illness), and admission details (type of admission, visitors, deposit). Target: length of stay, bucketed into 11 ranges.

## Key Findings
- Fixed a hidden Excel auto-date-conversion bug that had silently corrupted the "11-20" category into "20-Nov" in both `Stay` and `Age` columns.
- Confirmed via hypothesis testing (Chi-Square, Kruskal-Wallis) that Severity of Illness, Age, and Department all significantly affect Length of Stay.
- `Admission_Deposit` emerged as the top predictive feature in tree-based models despite showing almost no linear correlation with the target — revealing a non-linear relationship that linear models can't capture.

## Model Performance Summary

| Model | Accuracy | Weighted F1 |
|---|---|---|
| Logistic Regression | 24.59% | 0.2605 |
| **Random Forest (Final)** | **36.71%** | **0.3618** |
| XGBoost | 30.00% | 0.3107 |

Full details in [MODEL_COMPARISON_REPORT.md](./MODEL_COMPARISON_REPORT.md).

## Tech Stack
Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Seaborn · SciPy

## Author
Yaswanth Gunda — Individual contribution, Datamites capstone project.
