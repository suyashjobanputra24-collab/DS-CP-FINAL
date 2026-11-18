# PromiseGuard: Model Architecture, Training & Performance Documentation

**Platform**: PromiseGuard™ Enterprise Delivery Intelligence Engine  
**Artifact Directory**: `models/`  
**Dataset Split**: Chronological Split ($N_{\text{train}} = 31,908$, $N_{\text{val}} = 6,838$, $N_{\text{test}} = 6,838$)  

---

## 1. ETA Regression Models

### Model 1: Global Median Baseline
- **File**: Computed in `src/eta_model.py`
- **Description**: Naive baseline predicting the fixed training set median ($\hat{y} = 26.0$ min) for every order.
- **Val Performance**: MAE = 7.77 min | RMSE = 9.62 min | $R^2 = -0.0040$
- **Test Performance**: MAE = 7.48 min | RMSE = 9.31 min | $R^2 = -0.0005$

### Model 2: Segment Median Baseline
- **File**: Computed in `src/eta_model.py`
- **Description**: Lookup table predicting the median delivery duration by Road Traffic Density and Distance Quartile.
- **Val Performance**: MAE = 6.60 min | RMSE = 8.31 min | $R^2 = 0.2507$
- **Test Performance**: MAE = 6.48 min | RMSE = 8.14 min | $R^2 = 0.2355$

### Model 3: Ridge Linear Regression
- **File**: `models/ridge_eta_pipeline.joblib`
- **Hyperparameters**: $\alpha = 1.0$, `StandardScaler` on numerics, `OneHotEncoder` on categoricals.
- **Val Performance**: MAE = 4.83 min | RMSE = 6.09 min | $R^2 = 0.5972$
- **Test Performance**: MAE = 4.85 min | RMSE = 6.10 min | $R^2 = 0.5702$

### Model 4: Random Forest Regressor
- **File**: `models/rf_eta_pipeline.joblib`
- **Hyperparameters**: `n_estimators=100`, `max_depth=14`, `min_samples_split=10`, `random_state=42`.
- **Val Performance**: MAE = 3.29 min | RMSE = 4.15 min | $R^2 = 0.8132$
- **Test Performance**: MAE = 3.25 min | RMSE = 4.07 min | $R^2 = 0.8085$

### Model 5: HistGradientBoosting Regressor (Primary Production Model)
- **File**: `models/best_eta_pipeline.joblib` / `models/hist_gb_eta_pipeline.joblib`
- **Hyperparameters**: `max_iter=150`, `max_depth=8`, `learning_rate=0.08`, `random_state=42`.
- **Val Performance**: MAE = 3.24 min | RMSE = 4.08 min | $R^2 = 0.8193$
- **Test Performance**: MAE = 3.26 min | RMSE = 4.08 min | $R^2 = 0.8080$ | MedAE = 2.83 min

---

## 2. Uncertainty Quantification Models

### 2.1 Quantile Gradient Boosting
- **Artifacts**: `models/quantile_low_q05.joblib` (Pinball loss $\alpha = 0.05$) & `models/quantile_high_q95.joblib` ($\alpha = 0.95$)
- **Nominal Target Coverage**: 90.0%
- **Observed Empirical Test Coverage**: **87.34%**
- **Average Interval Width**: **12.74 minutes** (Median width: 12.55 min)
- **Miss Breakdown**: 6.62% arrived faster than lower bound; 6.04% arrived slower than upper bound.

### 2.2 Split-Conformal Prediction
- **Calibration Set**: Validation set ($N_{\text{cal}} = 6,838$)
- **Conformal Threshold**: $q_{\text{conf}} = \pm 6.52$ minutes (at 90.03% finite-sample rank)
- **Observed Empirical Test Coverage**: **90.33%**
- **Average Interval Width**: **13.04 minutes**
- **Miss Breakdown**: 4.17% below interval; 5.50% above interval.

---

## 3. Promise-Break Risk Classifiers

### Model: HistGradientBoosting Classifier (Class-Weighted & Calibrated)
- **Artifact**: `models/best_promise_risk_pipeline.joblib`
- **Target**: $Y_{\text{late}} \in \{0, 1\}$ (Base late rate: 6.43% on test set)
- **Hyperparameters**: `max_iter=150`, `max_depth=7`, `class_weight='balanced'`, `random_state=42`.
- **Test ROC-AUC**: **0.9829**
- **Test PR-AUC**: **0.8053**
- **Test Brier Score**: **0.0436** (Calibrated probability output)
- **Operational Optimal Threshold**: $T^* = 0.57$
  - Test Recall on Delayed Orders: **90.68%** (Captured 399 out of 440 delayed orders)
  - Test Precision: **54.14%**
  - Test F1-Score: **0.678**

---

## 4. Customer Satisfaction Classifiers (FoodHub)
- **Artifact**: `models/satisfaction_logistic_pipeline.joblib`
- **Dataset**: FoodHub ($N_{\text{rated}} = 1,162$)
- **Target**: Satisfied ($\ge 4$ stars) vs Dissatisfied ($\le 3$ stars)
- **Test ROC-AUC**: **0.4615**
- **Takeaway**: Confirms scientifically that delivery transit time alone cannot discriminate customer ratings without data on food temperature, order correctness, and kitchen quality.
