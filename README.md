# 🏥 Hospital Length of Stay Prediction — PRCP-1022

A machine learning project for predicting a patient's hospital length of stay using admission-time hospital, patient, and administrative information.

The project analyzes approximately 318,000 hospital admission records and frames the prediction task as an **11-class ordinal classification problem**, with length of stay represented by ranges from `"0-10"` days to `"More than 100 Days"`.

The project covers data cleaning, exploratory data analysis, statistical hypothesis testing, feature engineering, categorical encoding, model training, class-imbalance handling, hyperparameter tuning, model comparison, and production-model recommendation.

## 🎯 Problem Statement

The original project requirement asks for a model to predict a patient's length of hospital stay.

However, the dataset does not provide an exact number of stay days. Instead, the target variable `Stay` is divided into **11 ordered ranges**, such as:

* `0-10`
* `11-20`
* `21-30`
* ...
* `91-100`
* `More than 100 Days`

Because the actual target is bucketed and ordered, the project treats the problem as an **ordinal multi-class classification task** rather than ordinary regression.

The model uses information available around hospital admission to predict the expected length-of-stay category.

## 📊 Project Type

**Type:** Classification

**Problem:** 11-class ordinal classification

**Target:** `Stay`

The target classes have a natural order, from shorter stays to longer stays.

## 🗂️ Dataset

The dataset contains approximately **318,000 hospital admission records** with **18 original features**.

The features cover:

* Hospital information
* Department information
* Patient demographics
* Patient severity
* Admission details
* Administrative information
* Length of stay

### Important Features

Examples of the available attributes include:

* Hospital type
* Hospital region
* Department
* Ward type
* Ward facility
* Age
* City code
* Type of admission
* Severity of illness
* Visitors with patient
* Available extra rooms in hospital
* Admission deposit
* Stay

## 🔎 Exploratory Data Analysis

The project performs exploratory analysis to understand the relationship between hospital admission characteristics and length of stay.

The analysis includes:

* Univariate analysis
* Bivariate analysis
* Multivariate analysis
* Distribution analysis
* Categorical feature analysis
* Target-class distribution analysis
* Correlation analysis
* Feature importance analysis

### Key Findings

* **Severity of Illness** has a significant relationship with length of stay.
* **Age** significantly affects length of stay.
* **Department** also has a significant relationship with the target.
* `Admission_Deposit` emerged as the most important feature in the final Random Forest model.
* `Admission_Deposit` had very weak linear correlation with the target, demonstrating that important predictive relationships can be non-linear.
* The target variable is significantly imbalanced, with some stay categories containing far fewer records than the majority classes.

## 🧹 Data Cleaning & Preprocessing

### Hidden Excel Date-Conversion Error

During data inspection, an unexpected category called `20-Nov` was discovered in both the `Stay` and `Age` columns.

The expected category was:

`11-20`

This was identified as an Excel auto-formatting issue where the original `"11-20"` value had been interpreted as a date and converted to `"20-Nov"`.

The corrupted values were mapped back to `11-20` before further processing.

This step was important because leaving the corrupted value unchanged would have created an incorrect category and potentially affected ordinal encoding and model training.

### Target Representation

Two numerical representations were considered:

* **Ordinal encoding** — preserves the natural order of the stay categories.
* **Stay_days_est** — midpoint-based estimated stay duration for potential regression use.

The project uses **ordinal/multi-class classification as the primary approach**, because the actual ground truth consists of stay ranges rather than exact stay durations.

### Categorical Encoding

Categorical variables were converted into numerical representations so they could be processed by the machine learning algorithms.

The repository includes:

`label_encoders.joblib`

This stores the fitted encoders required for consistent preprocessing of future data.

## ⚖️ Class Imbalance

The `Stay` target is heavily imbalanced.

The largest target class is approximately **31.89 times larger** than the smallest class.

Instead of using SMOTE or other synthetic oversampling methods, class weighting was used.

### Techniques Used

**Logistic Regression**

`class_weight='balanced'`

**Random Forest**

`class_weight='balanced'`

**XGBoost**

Manually calculated balanced sample weights were used during training.

This approach avoids generating potentially unrealistic synthetic patient records while still giving minority classes greater importance during training.

## 🤖 Machine Learning Models

Three classification models were evaluated:

1. **Logistic Regression**
2. **Random Forest**
3. **XGBoost**

### Model Characteristics

| Model               | Approach                       |
| ------------------- | ------------------------------ |
| Logistic Regression | Linear classification baseline |
| Random Forest       | Tree-based bagging ensemble    |
| XGBoost             | Gradient-boosting ensemble     |

The models were compared using accuracy and weighted F1-score.

Because the target is significantly imbalanced, **weighted F1-score was treated as the primary model-selection metric**, while accuracy was reported as a secondary metric.

## 📈 Model Performance

| Model                     |   Accuracy | Weighted F1 |
| ------------------------- | ---------: | ----------: |
| Logistic Regression       |     24.59% |      0.2605 |
| **Random Forest (Final)** | **36.71%** |  **0.3618** |
| XGBoost                   |     30.00% |      0.3107 |

### Model Comparison

Random Forest achieved the strongest overall performance.

It produced:

* **Accuracy:** 36.71%
* **Weighted F1:** 0.3618

XGBoost achieved 30.00% accuracy and a weighted F1 of 0.3107, while Logistic Regression achieved 24.59% accuracy and a weighted F1 of 0.2605.

The results indicate that tree-based models are better suited to capturing the non-linear relationships present in the dataset.

## ⚙️ Hyperparameter Tuning

Hyperparameter tuning was performed using cross-validation.

### Random Forest

RandomizedSearchCV was used to tune the Random Forest model.

An initial tuning attempt produced a `MemoryError` because both the search process and the Random Forest model were configured to use all available CPU cores simultaneously.

This created excessive nested parallelism and memory consumption.

The configuration was corrected by:

* Setting the Random Forest's internal `n_jobs=1`
* Limiting the outer search to `n_jobs=4`

This controlled memory usage while still allowing parallel cross-validation.

### Tuning Result

After tuning:

* Accuracy: **36.71%**
* Weighted F1: **0.3618**

Although accuracy decreased slightly compared with the untuned model, weighted F1 improved from **0.3599 to 0.3618**, which was considered more important because of the class imbalance.

### XGBoost

XGBoost was also tuned and produced only a marginal improvement:

* Accuracy: **29.68% → 30.00%**
* Weighted F1: **0.3098 → 0.3107**

## 🏆 Final Model Recommendation

### Random Forest

The **tuned Random Forest** is recommended as the production candidate because it achieved the highest weighted F1-score and accuracy among the evaluated models.

**Final performance:**

* **Accuracy:** 36.71%
* **Weighted F1:** 0.3618

Additional advantages include:

* Handles non-linear relationships
* Does not require feature scaling
* Provides feature importance
* Works naturally with complex interactions
* Easier to interpret than many black-box alternatives

The most important features in the final Random Forest model were:

| Feature                             | Importance |
| ----------------------------------- | ---------: |
| `Admission_Deposit`                 |      0.186 |
| `Visitors_with_Patient`             |      0.170 |
| `Age_ordinal`                       |      0.102 |
| `City_Code_Patient`                 |      0.101 |
| `Available_Extra_Rooms_in_Hospital` |      0.067 |

## ⚠️ Important Limitations

The model's overall accuracy of approximately **37% across 11 classes is not production-grade precision**.

The model should therefore be considered a **decision-support signal rather than an authoritative clinical prediction**.

Additional limitations include:

* Rare stay classes remain difficult to predict.
* Middle-duration classes such as `41-50` and `61-70` days have limited training examples.
* The surgery department is severely underrepresented.
* Model performance is based on the available historical dataset.
* Predictions should be validated on new and representative hospital data before real-world deployment.
* Healthcare decisions should not rely solely on this model.

## 💡 Potential Applications

A length-of-stay prediction system could help hospitals with:

* Bed and room planning
* Resource allocation
* Staff scheduling
* Discharge planning
* Capacity management
* Identifying potentially long-stay patients
* Operational planning

The model should support healthcare professionals rather than replace clinical judgment.

## 📁 Repository Contents

| File                                      | Description                                                                 |
| ----------------------------------------- | --------------------------------------------------------------------------- |
| `HealthCareAnalytics.csv`                 | Hospital admission dataset                                                  |
| `Hospital Stay Prediction Solution.ipynb` | Complete analysis, preprocessing, modeling, evaluation, and tuning notebook |
| `label_encoders.joblib`                   | Saved categorical encoders for consistent preprocessing                     |
| `CHALLENGES_REPORT.md`                    | Detailed data and modeling challenges with techniques and reasoning         |
| `MODEL_COMPARISON_REPORT.md`              | Model comparison and final production recommendation                        |
| `LICENSE`                                 | Project license                                                             |
| `.gitignore`                              | Git ignored files                                                           |
| `README.md`                               | Project documentation                                                       |

## 🔄 Project Workflow

```text
Hospital Admission Dataset
          ↓
Data Understanding
          ↓
Data Quality Inspection
          ↓
Fix Hidden Data Corruption
          ↓
Exploratory Data Analysis
          ↓
Statistical Hypothesis Testing
          ↓
Feature Engineering
          ↓
Ordinal Encoding
          ↓
Handle Class Imbalance
          ↓
Train/Test Split
          ↓
Train Classification Models
          ↓
Model Evaluation
          ↓
Hyperparameter Tuning
          ↓
Model Comparison
          ↓
Random Forest Selection
```

## 🛠️ Tech Stack

**Programming Language**

* Python

**Libraries**

* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* SciPy
* Joblib

**Development Environment**

* Jupyter Notebook

## 🚀 Future Improvements

* Improve minority-class prediction
* Explore ordinal classification algorithms specifically designed for ordered targets
* Investigate calibrated class weighting
* Test alternative resampling strategies carefully
* Explore advanced boosting models
* Perform more extensive cross-validation
* Collect additional clinically relevant admission-time features
* Validate the model on data from other hospitals
* Deploy the model as a decision-support API or dashboard
* Add model explainability using SHAP or similar techniques

## 📌 Conclusion

This project developed a machine learning approach for predicting hospital length of stay using information available around admission.

Because the dataset provides length of stay as ordered ranges rather than exact days, the problem was framed as an **11-class ordinal classification problem**.

The project addressed several important challenges, including hidden Excel date corruption, severe target imbalance, non-linear feature relationships, and computational limitations during hyperparameter tuning.

Among the three evaluated models, the tuned **Random Forest** achieved the best overall performance with **36.71% accuracy and a weighted F1-score of 0.3618**.

However, the current performance is not sufficient for authoritative real-world clinical prediction. The model is better positioned as an **operational decision-support tool**, subject to further validation and improvement.

---

**Project:** Hospital Length of Stay Prediction
**Project ID:** PRCP-1022-HospitalStayPred
**Problem Type:** 11-Class Ordinal Classification
