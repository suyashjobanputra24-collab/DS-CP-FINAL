"""
PromiseGuard: Jupyter Notebook Generator
Generates clean, executable, well-commented Jupyter Notebooks for all 11 curriculum modules.
"""

import os
import nbformat as nbf


def make_notebook(title: str, description: str, code_cells: list, dest_path: str):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(f"# {title}\n\n**MD3135 Data Science Project: PromiseGuard**  \n**Authors**: Suyash Jobanputra & Shreyash Jobanputra (Project ID: MDM-DS-039)  \n**Guide**: Dr. Kapil N. Vhatkar  \n\n---\n\n{description}")
    ]
    for cell in code_cells:
        if isinstance(cell, tuple):
            nb.cells.append(nbf.v4.new_markdown_cell(cell[0]))
            nb.cells.append(nbf.v4.new_code_cell(cell[1]))
        else:
            nb.cells.append(nbf.v4.new_code_cell(cell))
            
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "w") as f:
        nbf.write(nb, f)
    print(f"Generated {dest_path}")


def generate_all_notebooks():
    nb_dir = "notebooks"
    os.makedirs(nb_dir, exist_ok=True)

    # 01_dataset_audit.ipynb
    make_notebook(
        "Module 1: Dataset Ingestion & Quality Audit",
        "Inspects schemas, detects incorrect types, missingness, impossible values, and generates automatic data audit summaries.",
        [
            ("### 1. Ingestion and Setup", "import os\nimport pandas as pd\nimport numpy as np\nfrom src.data_loader import DataLoader\n\nloader = DataLoader()\ndf_ops = loader.load_delivery_operations_raw()\ndf_fh = loader.load_foodhub_raw()\nprint(f'Delivery Operations Shape: {df_ops.shape}')\nprint(f'FoodHub Shape: {df_fh.shape}')"),
            ("### 2. Missing Value and Data Type Verification", "print('Delivery Operations Nulls:')\nprint(df_ops.isnull().sum()[df_ops.isnull().sum() > 0])\nprint('\\nFoodHub Rating Distribution:')\nprint(df_fh['rating'].value_counts())"),
            ("### 3. Generate Audit Summary", "audit_ops, audit_fh = loader.generate_audit_reports()\nprint('Audit JSONs and Markdown reports generated.')")
        ],
        f"{nb_dir}/01_dataset_audit.ipynb"
    )

    # 02_data_cleaning.ipynb
    make_notebook(
        "Module 2: Data Cleaning & Feature Engineering",
        "Implements Haversine distance calculations, GPS coordinate corrections, timestamp extractions, and temporal train/validation/test splits.",
        [
            ("### 1. Execute Data Cleaning Pipeline", "import pandas as pd\nfrom src.preprocessing import DeliveryDataPipeline, FoodHubPipeline\n\nops_pipe = DeliveryDataPipeline()\ntrain_df, val_df, test_df = ops_pipe.prepare_and_save_data()\n\nfh_pipe = FoodHubPipeline()\nfh_clean = fh_pipe.load_and_clean_raw()\n\nprint(f'Train orders: {len(train_df):,}, Val orders: {len(val_df):,}, Test orders: {len(test_df):,}')"),
            ("### 2. Feature Inspection", "print('Engineered Features:')\nprint(train_df[['distance_km', 'order_hour', 'is_peak_hour', 'pickup_prep_delay_min', 'promised_eta_min', 'is_promise_broken']].head())")
        ],
        f"{nb_dir}/02_data_cleaning.ipynb"
    )

    # 03_exploratory_analysis.ipynb
    make_notebook(
        "Module 3: Exploratory Data Analysis (EDA)",
        "Visualizes delivery time distributions, distance relationships, diurnals, traffic congestion, and customer satisfaction ratings.",
        [
            ("### 1. Run Complete EDA Pipeline", "from src.eda import EDAPipeline\n\neda = EDAPipeline()\neda.generate_all_plots()\nprint('19 EDA plots generated in reports/figures/')"),
            ("### 2. Inspect Target Variable Distribution", "import matplotlib.pyplot as plt\nimport seaborn as sns\nimport pandas as pd\n\ndf = pd.read_csv('data/processed/cleaned_delivery_operations.csv')\nprint(df['Time_taken (min)'].describe())\nplt.figure(figsize=(7, 4))\nsns.histplot(df['Time_taken (min)'], kde=True, bins=25, color='#1E3A8A')\nplt.title('Delivery Duration (minutes)')\nplt.show()")
        ],
        f"{nb_dir}/03_exploratory_analysis.ipynb"
    )

    # 04_statistical_analysis.ipynb
    make_notebook(
        "Module 4: Statistical Analysis & Hypothesis Testing",
        "Formulates and formally tests research hypotheses H1 to H5 (ANOVA, Kruskal-Wallis, Welch's t-test, Spearman rank correlation).",
        [
            ("### 1. Run Formal Inferential Hypothesis Testing", "from src.statistical_analysis import StatisticalInference\n\nstat = StatisticalInference()\nresults = stat.run_all_tests()\nprint('Hypothesis Tests Complete:')\nfor k, v in results.items():\n    print(f\"{k}: {v['decision']}\")"),
            ("### 2. Review Statistical Test Summary", "import json\nwith open('reports/tables/statistical_hypothesis_tests.json') as f:\n    print(json.dumps(json.load(f), indent=2)[:800])")
        ],
        f"{nb_dir}/04_statistical_analysis.ipynb"
    )

    # 05_eta_regression.ipynb
    make_notebook(
        "Module 5: ETA Regression Modeling & Baseline Comparison",
        "Trains and benchmarks Global Median, Segment Median, Ridge Regression, Random Forest, and HistGradientBoosting.",
        [
            ("### 1. Train and Benchmark Regressors", "from src.eta_model import ETAModelTrainer\n\ntrainer = ETAModelTrainer()\nresults, test_preds = trainer.train_and_evaluate_all()"),
            ("### 2. View Performance Comparison Table", "import pandas as pd\ncomp_df = pd.read_csv('reports/tables/model_comparison_table.csv')\nprint(comp_df)")
        ],
        f"{nb_dir}/05_eta_regression.ipynb"
    )

    # 06_prediction_reliability.ipynb
    make_notebook(
        "Module 6: Delivery Prediction Reliability & Error Slices",
        "Analyzes underprediction vs overprediction, large late errors, and segment-level MAE across traffic, weather, and distance.",
        [
            ("### 1. Run Error Slicing Audit", "from src.reliability import ReliabilityAnalyzer\n\nanalyzer = ReliabilityAnalyzer()\nsummary = analyzer.analyze_all()\nprint(f\"Overall MAE: {summary['overall_mae']} min | Severe late (>10m): {summary['severe_late_10m_rate_pct']}%\")"),
            ("### 2. Traffic Density Segment Performance", "import pandas as pd\nprint(pd.DataFrame(summary['segments']['traffic']).T)")
        ],
        f"{nb_dir}/06_prediction_reliability.ipynb"
    )

    # 07_uncertainty_analysis.ipynb
    make_notebook(
        "Module 7: Uncertainty-Aware ETA Estimation",
        "Implements Quantile Gradient Boosting (alpha=0.05, 0.95) and Split-Conformal Prediction, evaluating empirical coverage on held-out test data.",
        [
            ("### 1. Train Quantile Regressors & Conformal Calibration", "from src.uncertainty import UncertaintyEstimator\n\nestimator = UncertaintyEstimator()\nunc_res = estimator.train_and_evaluate(nominal_coverage=0.90)\nprint(f\"Quantile GB Coverage: {unc_res['quantile_gradient_boosting']['observed_coverage_pct']}%\")\nprint(f\"Split-Conformal Coverage: {unc_res['split_conformal_prediction']['observed_coverage_pct']}%\")"),
            ("### 2. Segment Coverage Audit", "import pandas as pd\nprint(pd.DataFrame(unc_res['segment_coverage']['traffic']).T)")
        ],
        f"{nb_dir}/07_uncertainty_analysis.ipynb"
    )

    # 08_promise_risk.ipynb
    make_notebook(
        "Module 8: Delivery Promise-Break Risk Modeling",
        "Calibrated classification models detecting when orders are likely to breach customer-facing delivery promises.",
        [
            ("### 1. Train and Evaluate Promise-Break Classifiers", "from src.promise_risk import PromiseRiskClassifier\n\nclf = PromiseRiskClassifier()\nrisk_res = clf.train_and_evaluate()\nfor m, vals in risk_res.items():\n    print(f\"{m}: ROC-AUC={vals['test_roc_auc']}, Recall={vals['test_recall']*100:.1f}%\")")
        ],
        f"{nb_dir}/08_promise_risk.ipynb"
    )

    # 09_customer_satisfaction.ipynb
    make_notebook(
        "Module 9: Customer Satisfaction Analytics",
        "Empirical study of genuine post-delivery customer ratings from FoodHub and statistical modeling of dissatisfaction.",
        [
            ("### 1. Run Customer Satisfaction Pipeline", "from src.satisfaction import SatisfactionAnalyzer\n\nsat = SatisfactionAnalyzer()\nsat_res = sat.run_analysis_and_models()\nprint(f\"Rated orders: {sat_res['rated_orders']}, Unrated: {sat_res['unrated_orders']} ({sat_res['unrated_percentage']}%\")\nprint('Logistic Regression on Dissatisfaction ROC-AUC:', sat_res['models']['logistic_regression']['roc_auc'])")
        ],
        f"{nb_dir}/09_customer_satisfaction.ipynb"
    )

    # 10_explainable_ml.ipynb
    make_notebook(
        "Module 10: Explainable Machine Learning (SHAP & Permutation)",
        "Calculates global permutation feature importance and local SHAP values to explain individual and dataset-level model predictions.",
        [
            ("### 1. Run Model Explainability Pipeline", "from src.explainability import ModelExplainer\n\nexplainer = ModelExplainer()\nexp_res = explainer.explain_all()\nprint('Top 5 Permutation Features:')\nfor f, v in list(exp_res['permutation_importance'].items())[:5]:\n    print(f\"{f}: +{v['mean_importance']} min\")")
        ],
        f"{nb_dir}/10_explainable_ml.ipynb"
    )

    # 11_intervention_simulation.ipynb
    make_notebook(
        "Module 11: Proactive Intervention Simulator",
        "Offline simulation comparing Fixed Promise, Risk-Threshold Alerting, Interval-Aware ETA, and Dispatch Operations Review.",
        [
            ("### 1. Simulate Operational Response Policies", "from src.intervention_simulator import InterventionSimulator\n\nsim = InterventionSimulator()\nsim_res = sim.simulate_all_policies(risk_threshold=0.57)\nimport pandas as pd\nprint(pd.DataFrame(sim_res['policies'])[['policy_name', 'alert_rate_pct', 'recall_pct', 'precision_pct', 'missed_late_orders']])")
        ],
        f"{nb_dir}/11_intervention_simulation.ipynb"
    )

if __name__ == "__main__":
    generate_all_notebooks()
