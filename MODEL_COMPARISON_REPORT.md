# Model Comparison Report — Hospital Length of Stay Prediction

## Project: PRCP-1022-HospitalStayPred

## Objective
Compare the performance of three machine learning models trained to predict a patient's hospital length of stay (bucketed into 11 ordinal classes), and recommend the best model for production deployment.

## Models Evaluated

| Model | Type | Class Imbalance Handling |
|---|---|---|
| Logistic Regression | Linear, baseline | `class_weight='balanced'` |
| Random Forest | Tree-based ensemble (bagging) | `class_weight='balanced'` |
| XGBoost | Tree-based ensemble (boosting) | Manual `sample_weight` (balanced) |

## Evaluation Metrics
Given the significant class imbalance in the target variable (~32:1 ratio between the largest and smallest Stay class), **weighted F1-score** was treated as the primary decision metric rather than raw accuracy, since accuracy alone can be misleadingly inflated by majority-class performance in imbalanced settings. Accuracy is reported alongside for interpretability.

## Results Summary

| Model | Accuracy (untuned) | F1 (untuned) | Accuracy (tuned) | F1 (tuned) |
|---|---|---|---|---|
| Logistic Regression | 24.59% | 0.2605 | 24.59% | 0.2605 |
| Random Forest | 37.83% | 0.3599 | 36.71% | **0.3618** |
| XGBoost | 29.68% | 0.3098 | 30.00% | 0.3107 |

## Key Findings

1. **Logistic Regression significantly underperformed** both tree-based models, and hyperparameter tuning produced zero improvement. This confirms the relationship between our features (particularly `Admission_Deposit`) and the target is fundamentally non-linear — a linear model cannot capture it regardless of tuning.

2. **Random Forest outperformed XGBoost**, which was an unexpected result given XGBoost's typical reputation as the stronger gradient-boosting algorithm. Investigation revealed this was linked to how class imbalance was handled: XGBoost's manually-applied `sample_weight` compounded more aggressively across its sequential boosting structure than Random Forest's native `class_weight='balanced'` did across its independent bagged trees, causing XGBoost to over-correct for minority classes at a steep cost to overall precision on majority classes.

3. **Hyperparameter tuning had a limited but directionally positive effect** on both tree-based models. For Random Forest, tuning slightly reduced accuracy but improved weighted F1, consistent with the search being optimized for `f1_weighted`. For XGBoost, tuning provided only marginal gains (<0.5% on both metrics), suggesting its underlying performance ceiling here is set more by the imbalance-handling approach than by hyperparameter configuration.

## Feature Importance (Final Random Forest Model)

The top predictive features were:
1. `Admission_Deposit` (0.186)
2. `Visitors_with_Patient` (0.170)
3. `Age_ordinal` (0.102)
4. `City_Code_Patient` (0.101)
5. `Available_Extra_Rooms_in_Hospital` (0.067)

Notably, `Admission_Deposit` ranked as the top feature despite showing near-zero linear correlation with the target (-0.05) in exploratory analysis — indicating it participates in non-linear feature interactions (likely with Department and Severity) that only tree-based models can detect.

## Recommendation

**Random Forest (tuned)** is recommended for production deployment. It achieved the best weighted F1-score (0.3618) and accuracy (36.71%) among all three models, showed the most balanced confusion matrix across both majority and minority classes, requires no feature scaling, and offers interpretable feature importance — an important practical advantage for healthcare stakeholders who need to understand the reasoning behind a prediction, not just receive a black-box output.

## Limitations
- Overall predictive performance (~37% accuracy on an 11-class problem) is far from production-grade precision and should be positioned as a **decision-support signal**, not an authoritative prediction, in any real deployment.
- Performance on rare middle classes (e.g., "41-50 days", "61-70 days") remains weak across all models due to limited training examples.
- The `surgery` department is severely underrepresented (~0.4% of records), so predictions involving that department carry additional uncertainty.
