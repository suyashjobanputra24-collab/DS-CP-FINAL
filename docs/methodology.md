# PromiseGuard™ Enterprise: Technical Methodology & Machine Learning Specifications

**Platform**: PromiseGuard™ Enterprise v3.2.0  
**Target Fleet Implementations**: Zomato • Swiggy • Zepto • Blinkit • DoorDash  
**Pipeline**: Dual Stream Inference (<24ms P99), 90% Conformal Calibration & Cost-Asymmetric SLA Classification ($T^* = 0.57$)  

---

## 1. Problem Formulation & End-to-End Pipeline

PromiseGuard is formulated as a multi-stage explainable machine learning and uncertainty quantification pipeline.

$$\text{Data Ingestion} \longrightarrow \text{Audit \& Preprocessing} \longrightarrow \text{Inference \& EDA} \longrightarrow \text{Regression (ETA)} \longrightarrow \text{Uncertainty (Conformal/Quantile)} \longrightarrow \text{Promise Risk} \longrightarrow \text{Explainability (SHAP)} \longrightarrow \text{Intervention Simulation}$$

---

## 2. Mathematical Foundations

### 2.1 Geographic Haversine Distance
Given restaurant coordinates $(\phi_1, \lambda_1)$ and delivery customer coordinates $(\phi_2, \lambda_2)$ in radians:

$$\Delta \phi = \phi_2 - \phi_1, \quad \Delta \lambda = \lambda_2 - \lambda_1$$

$$a = \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)$$

$$c = 2 \cdot \arcsin\left(\min(1, \sqrt{a})\right)$$

$$d = R \cdot c \quad \text{where } R = 6371.0 \text{ km}$$

### 2.2 Chronological Splitting Strategy
To guarantee zero look-ahead bias and eliminate temporal data leakage, orders are strictly sorted by date and time $t_i$:

$$\mathcal{D}_{\text{train}} = \{ (x_i, y_i) \mid t_i \le t_{70\%} \}$$
$$\mathcal{D}_{\text{val}} = \{ (x_i, y_i) \mid t_{70\%} < t_i \le t_{85\%} \}$$
$$\mathcal{D}_{\text{test}} = \{ (x_i, y_i) \mid t_i > t_{85\%} \}$$

All transformers, categorical encoders, and imputation statistics are fitted on $\mathcal{D}_{\text{train}}$ only:

$$\hat{\theta} = \text{Fit}(\mathcal{D}_{\text{train}}), \quad X_{\text{val}} \leftarrow \text{Transform}(X_{\text{val}}, \hat{\theta}), \quad X_{\text{test}} \leftarrow \text{Transform}(X_{\text{test}}, \hat{\theta})$$

---

### 2.3 ETA Regression Loss Functions & Models

1. **Global Median Baseline**:
   $$\hat{y}_{\text{base}} = \text{median}(y_{\text{train}})$$

2. **Ridge Regularized Linear Regression**:
   $$\min_{\mathbf{w}} \sum_{i=1}^N \left( y_i - \mathbf{w}^T x_i \right)^2 + \alpha \|\mathbf{w}\|_2^2$$

3. **Random Forest Regressor**:
   Ensemble of $B = 100$ decorrelated decision trees using bootstrap aggregating (bagging):
   $$\hat{y}_{\text{RF}}(x) = \frac{1}{B} \sum_{b=1}^B T_b(x)$$

4. **HistGradientBoosting Regressor**:
   Additive gradient tree boosting optimizing least absolute deviation (MAE) and squared error:
   $$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$

---

### 2.4 Uncertainty Quantification: Quantile Regression & Split-Conformal Prediction

#### A. Quantile Gradient Boosting
Trained using the asymmetric pinball loss at quantile level $\tau \in (0, 1)$:

$$\mathcal{L}_\tau(y, \hat{y}) = \begin{cases} 
\tau (y - \hat{y}) & \text{if } y \ge \hat{y} \\ 
(1 - \tau) (\hat{y} - y) & \text{if } y < \hat{y} 
\end{cases}$$

For nominal $1 - \alpha = 0.90$ coverage:
- Lower bound model: $\tau_1 = 0.05$
- Upper bound model: $\tau_2 = 0.95$
- Prediction Interval: $\mathcal{I}_{\text{quantile}}(x) = \left[ \hat{y}_{0.05}(x), \hat{y}_{0.95}(x) \right]$

#### B. Split-Conformal Prediction (Inductive Conformal Prediction)
Using validation set as calibration set $\mathcal{D}_{\text{cal}} = \{(x_j, y_j)\}_{j=1}^{n_{\text{cal}}}$:
1. Compute non-conformity absolute residuals:
   $$R_j = |y_j - \hat{\mu}(x_j)|$$
2. Compute the conformal calibration threshold $q_{\text{conf}}$ at level:
   $$p = \frac{\lceil (n_{\text{cal}} + 1)(1 - \alpha) \rceil}{n_{\text{cal}}}$$
   $$q_{\text{conf}} = \text{Quantile}(R_{1:n_{\text{cal}}}, p)$$
3. Conformal Prediction Interval:
   $$\mathcal{I}_{\text{conf}}(x) = \left[ \hat{\mu}(x) - q_{\text{conf}}, \, \hat{\mu}(x) + q_{\text{conf}} \right]$$
4. **Theoretical Guarantee**: Under exchangeability of calibration and test data:
   $$P\left( Y_{\text{test}} \in \mathcal{I}_{\text{conf}}(X_{\text{test}}) \right) \ge 1 - \alpha$$

---

### 2.5 Delivery Promise-Break Risk Modeling
- Stated SLA promise boundary:
  $$\text{Promise}_{\text{SLA}} = 15.0 + 2.2 \cdot d_{\text{km}} + \text{Buffer}_{\text{traffic}}$$
- Binary classification target:
  $$Y_{\text{late}} = \begin{cases} 1 & \text{if } \text{Actual Time} > \text{Promise}_{\text{SLA}} \\ 0 & \text{otherwise} \end{cases}$$
- Cost-sensitive decision threshold optimization:
  $$T^* = \arg\min_T \left[ 5 \cdot \text{FN}(T) + 1 \cdot \text{FP}(T) \right]$$

---

### 2.6 Explainability via SHAP (Shapley Additive exPlanations)
Local additive feature attribution decomposes model output $f(x)$ into:

$$f(x) = \phi_0 + \sum_{i=1}^M \phi_i(x)$$

where the Shapley value $\phi_i$ is computed across all feature subsets $S \subseteq F \setminus \{i\}$:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$

---

## 3. Evaluation Metrics Summary

| Metric | Formula | Target |
|---|---|---|
| **MAE** | $\frac{1}{N}\sum |y_i - \hat{y}_i|$ | Minimized |
| **RMSE** | $\sqrt{\frac{1}{N}\sum (y_i - \hat{y}_i)^2}$ | Minimized |
| **$R^2$** | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Maximized ($0 \le R^2 \le 1$) |
| **Empirical Coverage** | $\frac{1}{N}\sum \mathbb{I}(y_i \in [L_i, U_i])$ | Target $\approx 90.0\%$ |
| **Brier Score** | $\frac{1}{N}\sum (p_i - y_i)^2$ | Minimized ($0 \le BS \le 1$) |
| **PR-AUC** | $\sum_k (R_k - R_{k-1}) P_k$ | Maximized |
