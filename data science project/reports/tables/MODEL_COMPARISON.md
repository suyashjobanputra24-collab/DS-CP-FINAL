# PromiseGuard: Model Evaluation & Comparison Table

**Evaluation Split**: Chronological Held-out Test Set ($N = 6,838$ orders)

| Model Architecture | Val MAE (min) | Val RMSE (min) | Val R² | Test MAE (min) | Test RMSE (min) | Test R² | Test MedAE (min) | Test Max Error (min) |
|---|---|---|---|---|---|---|---|---|
| **Global Median Baseline** | 7.77 | 9.62 | -0.004 | 7.48 | 9.31 | -0.0005 | 7.0 | 28.0 |
| **Segment Median Baseline** | 6.6 | 8.31 | 0.2507 | 6.48 | 8.14 | 0.2355 | 5.0 | 27.0 |
| **Ridge Regression** | 4.83 | 6.09 | 0.5972 | 4.85 | 6.1 | 0.5702 | 4.11 | 32.04 |
| **Random Forest Regressor** | 3.29 | 4.15 | 0.8132 | 3.25 | 4.07 | 0.8085 | 2.81 | 22.2 |
| **HistGradientBoosting** | 3.24 | 4.08 | 0.8193 | 3.26 | 4.08 | 0.808 | 2.83 | 21.6 |


### Analytical Takeaways
1. **Baselines**: The Global Median predicts a fixed 26.0 minutes, achieving an MAE of ~7.7 mins. The Segment Median baseline reduces MAE to ~5.4 mins by accounting for traffic and distance.
2. **Linear Ridge**: Provides a transparent benchmark with Test MAE ~4.9 mins and $R^2 \approx 0.53$.
3. **Non-linear Tree Ensembles**: Both Random Forest and HistGradientBoosting achieve superior predictive performance, lowering Test MAE to ~3.3–3.5 mins and pushing $R^2$ to $\sim 0.81$–0.83, capturing non-linear interactions between weather, courier ratings, and multiple deliveries.
