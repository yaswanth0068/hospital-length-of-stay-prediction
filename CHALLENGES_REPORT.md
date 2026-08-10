# Challenges Report — Hospital Length of Stay Prediction

## Project: PRCP-1022-HospitalStayPred

This report documents the key data and modeling challenges encountered during this project, the techniques used to address them, and the reasoning behind each decision.

---

## 1. Hidden Data Corruption: Excel Auto-Date Conversion Bug

**Challenge:** During initial data exploration, the `Stay` and `Age` columns showed an unexpected category, `"20-Nov"`, which didn't fit the pattern of the other bucket labels (e.g., "0-10", "11-20", "21-30"...).

**Root Cause:** This is a classic Excel auto-formatting artifact. The original category label `"11-20"` was likely opened/saved in Microsoft Excel at some point, which automatically interpreted the day-month-style text as a date and silently converted it to `"20-Nov"` (Nov 20th).

**Technique Used & Reasoning:** We mapped `"20-Nov"` back to `"11-20"` in both affected columns using a simple `.replace()` operation before any further processing. This was essential to fix *before* creating ordinal encodings, since an unaddressed corruption would have either created a 12th nonsensical category or silently dropped/misplaced all "11-20" records from their correct ordinal position, corrupting both our EDA and downstream model training.

**Lesson:** Always inspect the *exact unique values* of categorical columns early, especially bucketed/range-style labels — this class of bug is common wherever CSV data has passed through Excel at any point in its lifecycle, and it's easy to miss since the corrupted value looks like a plausible category at a glance.

---

## 2. Target Variable Mismatch: Bucketed Categories vs. Continuous Regression Target

**Challenge:** The problem statement asks to "predict the length of stay in days" (implying regression), but the actual `Stay` column in the data is provided as 11 bucketed ranges (e.g., "0-10", "11-20", ..., "More than 100 Days"), not a continuous number.

**Technique Used & Reasoning:** We created two numeric representations: an **ordinal encoding** (`Stay_ordinal`, preserving bucket order for classification) and a **midpoint-based estimate** (`Stay_days_est`, approximating actual days for potential regression use). We proceeded with **ordinal/multi-class classification** as our primary modeling approach, since the ground truth itself is fundamentally bucketed — treating it as regression would require assuming a specific value within each bucket that isn't actually known, adding synthetic precision the data doesn't support.

**Lesson:** Always cross-check the problem statement's stated goal against what the raw data actually supports — a mismatch like this changes the entire modeling approach and evaluation strategy, and should be explicitly documented rather than silently resolved one way or another.

---

## 3. Significant Class Imbalance in the Target Variable

**Challenge:** The `Stay` target is heavily imbalanced — a 31.89:1 ratio between the largest class ("21-30 days") and the smallest ("61-70 days"). Standard model training would heavily favor majority classes and perform poorly on rarer but operationally important long-stay cases.

**Technique Used & Reasoning:** We used **class weighting** (`class_weight='balanced'` for Random Forest and Logistic Regression; manually computed `sample_weight` for XGBoost) rather than resampling techniques like SMOTE. With 11 overlapping ordinal classes and 250K+ training rows, generating synthetic samples across multiple minority classes simultaneously would have been computationally expensive and risked creating clinically implausible synthetic patient profiles — a real concern in a healthcare context.

**Lesson:** Class balancing isn't a one-size-fits-all fix — as seen with XGBoost, aggressive reweighting can overcorrect and hurt overall performance by trading majority-class precision for minority-class recall. The right degree of balancing depends on which errors matter most for the actual business use case, and should be evaluated empirically, not assumed.

---

## 4. Weak Linear Correlation Despite Strong Predictive Power (Admission_Deposit)

**Challenge:** Our correlation heatmap showed `Admission_Deposit` had almost no linear relationship with the target (-0.05), suggesting it might be a weak feature. However, Random Forest feature importance ranked it as the **single most important predictor** (0.19-0.30 importance depending on the model run).

**Technique Used & Reasoning:** We relied on **tree-based feature importance** (which captures non-linear relationships and interaction effects) rather than correlation coefficients alone (which only capture linear relationships) to judge true feature value. This also directly informed our model selection — it confirmed that linear models like Logistic Regression would be fundamentally limited on this dataset, which was validated by Logistic Regression's poor, tuning-resistant performance.

**Lesson:** Never rely solely on linear correlation to assess feature importance in a modeling context — a feature can be highly predictive through non-linear interactions that correlation coefficients completely miss. Always cross-check EDA-stage correlation findings against actual model-based feature importance before finalizing a feature set.

---

## 5. Computational/Memory Constraints During Hyperparameter Tuning

**Challenge:** Our first attempt at tuning Random Forest with `RandomizedSearchCV` threw a `MemoryError`, caused by nested parallelism — both the outer search (`n_jobs=-1`) and the inner model (`n_jobs=-1`) tried to use all available CPU cores simultaneously, spawning far more processes than the system could support, each holding a copy of the 250K-row dataset in memory.

**Technique Used & Reasoning:** We resolved this by keeping the inner model single-threaded (`n_jobs=1`) and limiting parallelism to the outer search only (`n_jobs=4`), which controlled total memory usage while still gaining meaningful speedup from parallel cross-validation folds.

**Lesson:** When combining scikit-learn's parallel search tools (`GridSearchCV`/`RandomizedSearchCV`) with inherently parallel models (Random Forest, XGBoost), always parallelize at only one level — nested `n_jobs=-1` settings multiply resource consumption in a way that can exhaust available memory even on moderately-sized datasets.
