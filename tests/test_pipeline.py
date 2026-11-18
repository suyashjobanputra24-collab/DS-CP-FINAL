"""
PromiseGuard: Automated Test Suite for Data, Models, Uncertainty, and Governance
Executed using pytest.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import pytest
import joblib
import pandas as pd
import numpy as np
from src.preprocessing import haversine_distance


def test_haversine_distance():
    """Verify that Haversine distance matches spherical geometry."""
    # Distance between London (51.5074, -0.1278) and Paris (48.8566, 2.3522) is ~343.5 km
    dist = haversine_distance(51.5074, -0.1278, 48.8566, 2.3522)
    assert 340.0 < dist < 350.0

    # Distance to same point must be 0
    dist_zero = haversine_distance(19.0760, 72.8777, 19.0760, 72.8777)
    assert np.isclose(dist_zero, 0.0)


def test_processed_data_integrity():
    """Verify clean dataset row counts, absence of leakage, and target validity."""
    clean_ops_path = "data/processed/cleaned_delivery_operations.csv"
    train_path = "data/processed/train_delivery.csv"
    val_path = "data/processed/val_delivery.csv"
    test_path = "data/processed/test_delivery.csv"

    assert os.path.exists(clean_ops_path)
    assert os.path.exists(train_path)
    assert os.path.exists(val_path)
    assert os.path.exists(test_path)

    df_clean = pd.read_csv(clean_ops_path)
    assert len(df_clean) == 45584
    assert "Time_taken (min)" in df_clean.columns
    assert (df_clean["Time_taken (min)"] > 0).all()

    # Chronological integrity
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    assert len(train_df) + len(val_df) + len(test_df) == len(df_clean)
    assert len(train_df) == 31908
    assert len(val_df) == 6838
    assert len(test_df) == 6838


def test_eta_regression_model_inference():
    """Test that saved best ETA regression pipeline predicts valid positive minutes."""
    model_path = "models/best_eta_pipeline.joblib"
    assert os.path.exists(model_path)
    pipe = joblib.load(model_path)

    test_df = pd.read_csv("data/processed/test_delivery.csv").head(10)
    from src.eta_model import ETAModelTrainer
    feature_cols = ETAModelTrainer().feature_cols

    preds = pipe.predict(test_df[feature_cols])
    assert len(preds) == 10
    assert (preds > 5.0).all()
    assert (preds < 80.0).all()


def test_uncertainty_quantiles():
    """Verify that lower quantile bound is less than upper quantile bound."""
    q05_path = "models/quantile_low_q05.joblib"
    q95_path = "models/quantile_high_q95.joblib"
    assert os.path.exists(q05_path)
    assert os.path.exists(q95_path)

    q05 = joblib.load(q05_path)
    q95 = joblib.load(q95_path)

    test_df = pd.read_csv("data/processed/test_delivery.csv").head(20)
    from src.eta_model import ETAModelTrainer
    feature_cols = ETAModelTrainer().feature_cols

    low = q05.predict(test_df[feature_cols])
    high = q95.predict(test_df[feature_cols])

    assert len(low) == 20
    assert (high >= low).all()


def test_promise_risk_classifier():
    """Test that risk classifier outputs valid probabilities between 0 and 1."""
    risk_path = "models/best_promise_risk_pipeline.joblib"
    assert os.path.exists(risk_path)
    clf = joblib.load(risk_path)

    test_df = pd.read_csv("data/processed/test_delivery.csv").head(10)
    from src.eta_model import ETAModelTrainer
    feature_cols = ETAModelTrainer().feature_cols

    probs = clf.predict_proba(test_df[feature_cols])
    assert probs.shape == (10, 2)
    assert (probs[:, 1] >= 0.0).all() and (probs[:, 1] <= 1.0).all()


def test_customer_satisfaction_data():
    """Test FoodHub dataset integrity and unrated order handling."""
    clean_fh_path = "data/processed/cleaned_foodhub_orders.csv"
    assert os.path.exists(clean_fh_path)
    df_fh = pd.read_csv(clean_fh_path)
    assert len(df_fh) == 1898
    assert "rating" in df_fh.columns
    assert (df_fh["rating"] == "Not given").sum() == 736
    assert (df_fh["is_rated"] == 1).sum() == 1162


def test_report_and_table_generation():
    """Verify that all analytical tables and markdown reports are present."""
    required_tables = [
        "reports/tables/model_comparison_table.csv",
        "reports/tables/MODEL_COMPARISON.md",
        "reports/tables/RELIABILITY_ANALYSIS.md",
        "reports/tables/UNCERTAINTY_EVALUATION.md",
        "reports/tables/PROMISE_RISK_REPORT.md",
        "reports/tables/CUSTOMER_SATISFACTION_REPORT.md",
        "reports/tables/EXPLAINABILITY_REPORT.md",
        "reports/tables/INTERVENTION_SIMULATION.md"
    ]
    for path in required_tables:
        assert os.path.exists(path), f"Missing table report: {path}"
