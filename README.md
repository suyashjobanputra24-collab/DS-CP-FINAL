# 🛡️ PromiseGuard: An Explainable Machine Learning Framework for Food Delivery ETA Reliability, Customer Dissatisfaction Analytics and Proactive Delay Prevention

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![Course: MD3135 Data Science](https://img.shields.io/badge/Course-MD3135%20Data%20Science-8A2BE2.svg)]()

> **Third Year Computer Engineering Capstone Project | Academic Year 2026–27**  
> **Project ID**: MDM-DS-039  
> **Student Authors**: Suyash Jobanputra & Shreyash Jobanputra  
> **Project Guide**: Dr. Kapil N. Vhatkar  

---

## 📌 Executive Summary & Problem Statement

Food-delivery aggregators (such as Zomato, Swiggy, Uber Eats, and Meituan) communicate an Estimated Time of Arrival (ETA) to consumers immediately upon checkout. However, presenting a single deterministic point estimate (e.g. *"Your order will arrive in 35 minutes"*) conceals multi-stage operational uncertainties. 

Two distinct orders can both be assigned a 35-minute ETA:
- **Order A**: Arrives in 36 minutes (predictable transit, negligible customer friction).
- **Order B**: Arrives in 54 minutes (severe unexpected delay, kitchen queue, or rain congestion).

When predictions fail, customers face meal disruption, platforms encounter spikes in customer support inquiries and refund costs, and couriers face unfair pressure.

**PromiseGuard** is an open, transparent, and reproducible Data Science research framework that addresses this challenge across five integrated layers:
1. **ETA Duration Prediction**: Accurately estimating delivery duration while benchmarking against simple baselines.
2. **Prediction Uncertainty Quantification**: Bounding estimates with calibrated **90% prediction intervals** via Quantile Gradient Boosting and Split-Conformal Prediction.
3. **Promise-Break Risk Modeling**: Calibrated classification predicting whether an order will breach its stated delivery commitment.
4. **Explainable AI (XAI)**: Deconstructing risk drivers and predictions using global and local **SHAP (Shapley Additive exPlanations)**.
5. **Customer Satisfaction Analytics**: Empirically evaluating genuine post-delivery ratings from the FoodHub real-world dataset.
6. **Proactive Intervention Policy Simulation**: Offline evaluation comparing operational response policies and alert thresholds.

---

## 📊 Key Measured Results & Benchmarks

All metrics below are calculated on a strictly **chronological held-out test set ($N = 6,838$ orders)** with zero future data leakage:

| Analytical Dimension | Benchmark Baseline | PromiseGuard Model | Measured Result | Operational Improvement |
|---|---|---|---|---|
| **ETA Regression (MAE)** | Global Median: 7.48 min | **HistGradientBoosting** | **3.26 min** | **56.4% Error Reduction** ($R^2 = 0.808$) |
| **ETA Regression (RMSE)**| Global Median: 9.31 min | **HistGradientBoosting** | **4.08 min** | **56.2% Variance Reduction** |
| **Prediction Interval Coverage** | Target: 90.0% | **Split-Conformal Prediction** | **90.33%** | **Mathematically Valid Coverage** ($\pm 6.52$ min) |
| **Quantile Prediction Interval** | Target: 90.0% | **Quantile Gradient Boosting** | **87.34%** | **Adaptive Width** (Mean width: 12.74 min) |
| **Promise-Break Risk ROC-AUC** | Random: 0.500 | **Calibrated HistGB Classifier** | **0.9829** | **Strong Discriminative Power** |
| **Promise-Break Risk PR-AUC** | Base Rate: 0.064 (6.4%)| **Calibrated HistGB Classifier** | **0.8053** | **High Precision on Imbalanced Late Class** |
| **Delay Interception Recall** | Policy A: 0.0% (0 caught)| **Policy B ($T^* = 0.57$)** | **90.68%** | **Caught 399 of 440 delays prior to breach** |
| **FoodHub Satisfaction ROC-AUC**| Chance: 0.500 | **Logistic Regression** | **0.4615** | **Proves transit time alone cannot predict ratings** |

---

## 🏗️ Project Architecture & Modules

```
PromiseGuard/
├── README.md                          # Master Project Documentation & Quickstart
├── BUILD_STATUS.md                    # Module-by-Module Progress & Phase Tracker
├── requirements.txt                   # Complete Python Package Dependencies
├── .gitignore                         # Git exclusion rules
│
├── data/
│   ├── raw/                           # Original immutable source benchmarks
│   │   ├── delivery_operations.csv    # 45,584 Indian delivery logistics records
│   │   └── foodhub_order.csv          # 1,898 FoodHub order experience records
│   ├── processed/                     # Cleaned datasets & temporal splits
│   │   ├── train_delivery.csv         # Chronological Train Split (70%, N=31,908)
│   │   ├── val_delivery.csv           # Chronological Validation Split (15%, N=6,838)
│   │   ├── test_delivery.csv          # Chronological Test Split (15%, N=6,838)
│   │   ├── test_with_promise_risk.csv # Test set with predictions, intervals & risk
│   │   └── cleaned_foodhub_orders.csv # Cleaned FoodHub dataset with ratings
│   └── external/
│       └── survey_instrument.json     # Primary customer experience survey schema
│
├── notebooks/                         # 11 Executable Step-by-Step Jupyter Notebooks
│   ├── 01_dataset_audit.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   ├── 04_statistical_analysis.ipynb
│   ├── 05_eta_regression.ipynb
│   ├── 06_prediction_reliability.ipynb
│   ├── 07_uncertainty_analysis.ipynb
│   ├── 08_promise_risk.ipynb
│   ├── 09_customer_satisfaction.ipynb
│   ├── 10_explainable_ml.ipynb
│   └── 11_intervention_simulation.ipynb
│
├── src/                               # Modular Python Data Science Engine
│   ├── data_loader.py                 # Dataset ingestion & automated quality audit
│   ├── preprocessing.py               # Coordinate cleaning, Haversine, temporal splits
│   ├── eda.py                         # Complete suite of 19 publication-grade figures
│   ├── statistical_analysis.py        # Formal hypothesis tests (H1-H5, ANOVA, Welch's t)
│   ├── eta_model.py                   # Baseline, Ridge, RF, HistGB regression pipelines
│   ├── reliability.py                 # Error slicing by traffic, weather, distance
│   ├── uncertainty.py                 # Quantile Gradient Boosting & Conformal Prediction
│   ├── promise_risk.py                # Calibrated SLA breach classifier & cost threshold
│   ├── satisfaction.py                # Genuine customer rating analytics & limitations
│   ├── explainability.py              # Permutation Importance & SHAP TreeExplainer
│   └── intervention_simulator.py      # Offline policy simulator (4 operational policies)
│
├── models/                            # Trained & Serialized ML Pipelines (.joblib)
│   ├── best_eta_pipeline.joblib       # Primary HistGradientBoosting Regressor
│   ├── rf_eta_pipeline.joblib         # Random Forest Regressor
│   ├── ridge_eta_pipeline.joblib      # Regularized Linear Ridge Pipeline
│   ├── quantile_low_q05.joblib        # Quantile Regressor (alpha = 0.05)
│   ├── quantile_high_q95.joblib       # Quantile Regressor (alpha = 0.95)
│   ├── best_promise_risk_pipeline.joblib # Calibrated Risk Classifier
│   └── shap_explainer_bundle.joblib   # Serialized SHAP TreeExplainer & sample space
│
├── reports/
│   ├── figures/                       # High-resolution (300 DPI) analysis charts (19+ figures)
│   ├── tables/                        # Formatted CSV, JSON, and Markdown analytical tables
│   ├── data_audit/                    # Data quality audit reports & JSON schemas
│   └── final_report/                  # Academic Capstone Project Report (Markdown)
│
├── app/
│   └── app.py                         # 10-Page Interactive Streamlit Application
│
├── tests/
│   └── test_pipeline.py               # Automated pytest verification test suite
│
└── docs/                              # Comprehensive Academic Documentation
    ├── viva_preparation.md            # Beginner-friendly guide & 25 viva questions/answers
    ├── methodology.md                 # Mathematical derivations, formulas & algorithms
    ├── data_dictionary.md             # Complete feature schemas, types & allowed ranges
    ├── model_documentation.md         # Hyperparameters, architectures & model cards
    ├── limitations.md                 # Academic constraints, ethics & worker safety
    ├── references.md                  # Academic citations (IEEE/APA)
    └── industry_research.md           # Public disclosures of Zomato, Swiggy, Meituan
```

---

## 🚀 Quickstart & Installation Guide

The project is built in Python 3.10+ and tested on macOS (Apple Silicon M-Series / Intel) and Linux.

### 1. Clone & Set Up Environment
```bash
# Clone the repository (or navigate to workspace directory)
cd "data science project"

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the End-to-End Data Science Pipeline
You can re-execute the entire pipeline from scratch or execute individual modules:
```bash
# 1. Run Data Ingestion & Quality Audit
python src/data_loader.py

# 2. Run Data Cleaning & Chronological Splitting
python src/preprocessing.py

# 3. Generate Exploratory Visualizations
python src/eda.py

# 4. Run Statistical Hypothesis Testing (H1 to H5)
python src/statistical_analysis.py

# 5. Train & Evaluate ETA Regression Models
python src/eta_model.py

# 6. Run Prediction Reliability & Error Slicing
python src/reliability.py

# 7. Train Uncertainty Estimators (Quantile & Conformal)
python src/uncertainty.py

# 8. Train Promise-Break Risk Classifiers
python src/promise_risk.py

# 9. Run Customer Satisfaction Analytics
python src/satisfaction.py

# 10. Compute SHAP & Feature Permutation Explanations
python src/explainability.py

# 11. Run Proactive Intervention Policy Simulator
python src/intervention_simulator.py
```

### 3. Launch the Interactive Website (Primary Web Platform)
```bash
python run_website.py
# or double-click start_website.bat (Windows)
```
Open your browser at **`http://127.0.0.1:8000`** to access the complete multi-page web platform:
- **Overview & Executive Dashboard**: KPI cards, problem formulation, and 5-layer framework architecture.
- **Live ETA & Uncertainty Predictor**: Real-time slider controls, point estimates, 90% quantile & conformal prediction intervals, SLA boundary timeline gauge, and proactive delay recommendations.
- **Operational Reliability & Slices**: Traffic, weather, and distance band MAE/RMSE comparisons and directional error balance.
- **Uncertainty Quantification**: Quantile Gradient Boosting vs Split-Conformal coverage breakdown.
- **Promise-Break Risk Classifier**: Benchmark comparison, ROC/PR curves, and cost-optimal threshold ($T^* = 0.57$).
- **Customer Satisfaction Study**: Real-world FoodHub rating distribution, negative result analysis, and multi-factorial discussion.
- **Explainable AI (SHAP & Permutation)**: Permutation importance ranking and SHAP summary beeswarm plot.
- **Proactive Intervention Simulator**: Dynamic interactive slider ($T \in [0.10, 0.90]$) with real-time recalculation of alerts, recall %, precision %, and cost index.
- **Dataset Audit & Explorer**: Full data dictionary, cleaning log, and test set sample record viewer.
- **Viva Voce Defense Portal**: Top 10 examiner questions with expandable model answers and 5-minute presentation guide.
- **Interactive REST API**: Live OpenAPI Swagger documentation at **`http://127.0.0.1:8000/docs`**.

### 4. Alternative Streamlit Dashboard (Optional)
If preferred for Python-specific demonstrations, the Streamlit app is also available:
```bash
python -m streamlit run app/app.py
```
Accessible at `http://localhost:8501`.

### 5. Run Automated Test Suite
```bash
python -m pytest tests/ -v
```
All 12 pipeline, data integrity, and web route tests will execute and verify the system end-to-end.

---

## 🎓 University Viva Voce & Demonstration Guide

For students **Suyash Jobanputra** and **Shreyash Jobanputra** defending this project during university evaluation:

1. **Which file should you read first?**  
   Read [`docs/viva_preparation.md`](docs/viva_preparation.md). It explains every single term from absolute zero, assumes no prior expertise, and includes 25+ real examiner questions with model answers.
2. **What should you demonstrate to the faculty?**
   - **Step 1**: Open the Streamlit dashboard on **Page 1 (Project Overview)** and state the problem of point-estimate uncertainty.
   - **Step 2**: Navigate to **Page 4 (Live ETA & Uncertainty Prediction)**. Move the distance and traffic sliders to demonstrate how the **90% Quantile Prediction Interval** widens under Jam traffic.
   - **Step 3**: Navigate to **Page 5 (Prediction Reliability Analytics)** and show the 8-fold increase in severe late errors under Jam traffic.
   - **Step 4**: Navigate to **Page 7 (Customer Satisfaction Analytics)** and explain the scientific negative finding: why delivery duration alone cannot predict customer ratings ($\text{ROC-AUC} \approx 0.46$).
   - **Step 5**: Navigate to **Page 9 (Proactive Intervention Simulator)** and show how threshold $T^* = 0.57$ catches 90.7% of delayed orders.
   - **Step 6**: Conclude with academic integrity and ethical safeguards from [`docs/limitations.md`](docs/limitations.md).

---

## 📜 Academic Integrity, Governance & Licensing
- **Licensing**: Academic non-commercial research use.
- **Data Governance**: No personal identifiers or customer phone numbers are collected or retained. Coordinates are aggregated for analytics.
- **Worker Fairness Safeguard**: Model predictions must never be used to penalize delivery couriers or encourage unsafe driving. Intervention policies recommend early customer communication and kitchen readiness reviews, not driver pressure.
