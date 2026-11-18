"""
PromiseGuard™ Enterprise Web Application Server
Powered by FastAPI, Uvicorn & Jinja2
Enterprise Delivery SLA Defense, Predictive ETA Intelligence & Churn Prevention Engine
Commercial B2B SaaS for High-Volume Food Delivery & Quick-Commerce Aggregators
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Request, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Ensure root directory is in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

app = FastAPI(
    title="PromiseGuard™ Enterprise | Delivery SLA Defense & Real-Time ETA Intelligence Engine",
    description="Autonomous Machine Learning Platform for Food Delivery ETA Reliability, Conformal Uncertainty Calibration & Proactive Delay SLA Defense",
    version="3.2.0-enterprise"
)

# Static and Templates
STATIC_DIR = os.path.join(ROOT_DIR, "website", "static")
TEMPLATES_DIR = os.path.join(ROOT_DIR, "website", "templates")
FIGURES_DIR = os.path.join(ROOT_DIR, "reports", "figures")
TABLES_DIR = os.path.join(ROOT_DIR, "reports", "tables")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if os.path.exists(FIGURES_DIR):
    app.mount("/reports/figures", StaticFiles(directory=FIGURES_DIR), name="figures")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ------------------------------------------------------------------------------
# Data & Model Preloading
# ------------------------------------------------------------------------------
eta_pipe = None
q05_pipe = None
q95_pipe = None
risk_pipe = None
cached_reports = {}
sample_orders = []

def load_system_assets():
    global eta_pipe, q05_pipe, q95_pipe, risk_pipe, cached_reports, sample_orders
    try:
        eta_pipe = joblib.load(os.path.join(MODELS_DIR, "best_eta_pipeline.joblib"))
        q05_pipe = joblib.load(os.path.join(MODELS_DIR, "quantile_low_q05.joblib"))
        q95_pipe = joblib.load(os.path.join(MODELS_DIR, "quantile_high_q95.joblib"))
        risk_pipe = joblib.load(os.path.join(MODELS_DIR, "best_promise_risk_pipeline.joblib"))
    except Exception as e:
        print(f"Warning: Model loading error: {e}")

    report_files = {
        "eta": "eta_regression_metrics.json",
        "reliability": "reliability_analysis.json",
        "uncertainty": "uncertainty_evaluation.json",
        "risk": "promise_risk_evaluation.json",
        "simulation": "intervention_simulation.json",
        "satisfaction": "customer_satisfaction_analysis.json",
        "explainability": "model_explainability.json",
        "hypothesis": "statistical_hypothesis_tests.json"
    }
    for key, filename in report_files.items():
        path = os.path.join(TABLES_DIR, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cached_reports[key] = json.load(f)

    # Load 50 sample records from test set for exploration
    test_path = os.path.join(DATA_DIR, "test_with_promise_risk.csv")
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        sample_orders = df_test.head(50).to_dict(orient="records")

load_system_assets()

# ------------------------------------------------------------------------------
# Pydantic Schemas for API
# ------------------------------------------------------------------------------
class OrderPredictionInput(BaseModel):
    distance_km: float = Field(7.5, ge=0.5, le=35.0, description="Delivery distance in km")
    Delivery_person_Age: float = Field(28.0, ge=18.0, le=60.0, description="Courier age")
    Delivery_person_Ratings: float = Field(4.7, ge=1.0, le=5.0, description="Courier historical rating")
    multiple_deliveries: float = Field(0.0, ge=0.0, le=3.0, description="Concurrent orders assigned")
    pickup_prep_delay_min: float = Field(10.0, ge=0.0, le=60.0, description="Restaurant pickup prep delay")
    order_hour: int = Field(20, ge=0, le=23, description="Order placement hour")
    Vehicle_condition: int = Field(1, ge=0, le=2, description="Vehicle condition (0=Poor, 1=Average, 2=Good)")
    Weather_conditions: str = Field("Sunny", description="Weather condition")
    Road_traffic_density: str = Field("Medium", description="Traffic density: Low, Medium, High, Jam")
    Type_of_order: str = Field("Meal", description="Type of order: Meal, Snack, Drinks, Buffet")
    Type_of_vehicle: str = Field("motorcycle", description="Vehicle type")
    Festival: str = Field("No", description="Festival active: Yes or No")
    City: str = Field("Metropolitian", description="City Urban Tier: Metropolitian, Urban, Semi-Urban")


class SimulationInput(BaseModel):
    threshold: float = Field(0.57, ge=0.05, le=0.95, description="Decision threshold T")


def run_prediction_pipeline(data: dict) -> dict:
    # Prepare DataFrame
    order_hour = int(data.get("order_hour", 20))
    distance = float(data.get("distance_km", 7.5))
    traffic = str(data.get("Road_traffic_density", "Medium"))
    weather = str(data.get("Weather_conditions", "Sunny"))

    df = pd.DataFrame([{
        "distance_km": distance,
        "Delivery_person_Age": float(data.get("Delivery_person_Age", 28)),
        "Delivery_person_Ratings": float(data.get("Delivery_person_Ratings", 4.7)),
        "multiple_deliveries": float(data.get("multiple_deliveries", 0.0)),
        "pickup_prep_delay_min": float(data.get("pickup_prep_delay_min", 10.0)),
        "order_hour": order_hour,
        "Vehicle_condition": int(data.get("Vehicle_condition", 1)),
        "is_weekend": 1 if order_hour >= 18 else 0,
        "is_peak_hour": 1 if order_hour in [12, 13, 14, 19, 20, 21, 22] else 0,
        "Weather_conditions": weather,
        "Road_traffic_density": traffic,
        "Type_of_order": str(data.get("Type_of_order", "Meal")),
        "Type_of_vehicle": str(data.get("Type_of_vehicle", "motorcycle")),
        "Festival": str(data.get("Festival", "No")),
        "City": str(data.get("City", "Metropolitian"))
    }])

    pred_eta = float(eta_pipe.predict(df)[0])
    pred_q_low = float(q05_pipe.predict(df)[0])
    pred_q_high = float(q95_pipe.predict(df)[0])
    pred_q_high = max(pred_q_high, pred_q_low + 2.0)

    # Conformal interval (fixed empirical calibrated margin ±6.52 min)
    conf_low = max(5.0, pred_eta - 6.52)
    conf_high = pred_eta + 6.52

    # Communicated SLA baseline promise
    traffic_buffer = {"Low": 2.0, "Medium": 6.0, "High": 12.0, "Jam": 18.0}.get(traffic, 6.0)
    promised_sla = round(15.0 + (distance * 2.2) + traffic_buffer, 1)

    # Promise-break risk
    risk_prob = float(risk_pipe.predict_proba(df)[0, 1])

    # Determine risk category & recommended proactive action
    if risk_prob >= 0.57:
        risk_level = "Severe Delay Risk"
        risk_class = "danger"
        recommendation = "PROACTIVE DISPATCH ACTION: Trigger instant delay notification to customer with updated ETA range (" + f"{int(pred_q_low)}–{int(pred_q_high+4)} min) and offer ₹50 loyalty credit before SLA is breached."
    elif risk_prob >= 0.30:
        risk_level = "Elevated Uncertainty"
        risk_class = "warning"
        recommendation = "ADAPTIVE COMMUNICATION: Display uncertainty-aware time range (" + f"{int(conf_low)}–{int(conf_high)} min) on tracking screen rather than single point estimate."
    else:
        risk_level = "On-Time Delivery"
        risk_class = "success"
        recommendation = "STANDARD TRACKING: Delivery parameters within normal operational bounds. Display standard countdown timer."

    # Identify top contributors
    contributors = []
    if traffic in ["High", "Jam"]:
        contributors.append(f"Heavy congestion ({traffic} traffic adds ~6–12 min delay)")
    if weather in ["Stormy", "Sandstorms", "Fog"]:
        contributors.append(f"Adverse weather conditions ({weather} induces high transit variance)")
    if float(data.get("multiple_deliveries", 0.0)) >= 1.0:
        contributors.append(f"Courier assigned {int(float(data.get('multiple_deliveries')))} concurrent orders")
    if float(data.get("pickup_prep_delay_min", 10.0)) >= 15:
        contributors.append("Prolonged kitchen prep delay (>15 min)")
    if distance >= 12.0:
        contributors.append(f"Long-haul delivery distance ({distance:.1f} km)")
    if not contributors:
        contributors.append("Normal operational transit conditions")

    return {
        "predicted_eta": round(pred_eta, 1),
        "pred_q_low": round(pred_q_low, 1),
        "pred_q_high": round(pred_q_high, 1),
        "conf_low": round(conf_low, 1),
        "conf_high": round(conf_high, 1),
        "promised_sla": round(promised_sla, 1),
        "risk_probability": round(risk_prob, 4),
        "risk_percentage": round(risk_prob * 100, 1),
        "risk_level": risk_level,
        "risk_class": risk_class,
        "recommendation": recommendation,
        "contributors": contributors,
        "interval_width_quantile": round(pred_q_high - pred_q_low, 1),
        "interval_width_conformal": round(conf_high - conf_low, 1)
    }

# ------------------------------------------------------------------------------
# Web Route Handlers
# ------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={
        "active_page": "home",
        "reports": cached_reports
    })


@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    default_inputs = {
        "distance_km": 7.5,
        "Delivery_person_Age": 28,
        "Delivery_person_Ratings": 4.7,
        "multiple_deliveries": 0.0,
        "pickup_prep_delay_min": 10.0,
        "order_hour": 20,
        "Vehicle_condition": 1,
        "Weather_conditions": "Sunny",
        "Road_traffic_density": "Medium",
        "Type_of_order": "Meal",
        "Type_of_vehicle": "motorcycle",
        "Festival": "No",
        "City": "Metropolitian"
    }
    initial_prediction = run_prediction_pipeline(default_inputs)
    return templates.TemplateResponse(request=request, name="predict.html", context={
        "active_page": "predict",
        "inputs": default_inputs,
        "prediction": initial_prediction
    })


@app.get("/reliability", response_class=HTMLResponse)
async def reliability_page(request: Request):
    return templates.TemplateResponse(request=request, name="reliability.html", context={
        "active_page": "reliability",
        "reliability": cached_reports.get("reliability", {}),
        "eta": cached_reports.get("eta", {})
    })


@app.get("/uncertainty", response_class=HTMLResponse)
async def uncertainty_page(request: Request):
    return templates.TemplateResponse(request=request, name="uncertainty.html", context={
        "active_page": "uncertainty",
        "uncertainty": cached_reports.get("uncertainty", {})
    })


@app.get("/promise-risk", response_class=HTMLResponse)
async def promise_risk_page(request: Request):
    return templates.TemplateResponse(request=request, name="risk.html", context={
        "active_page": "risk",
        "risk": cached_reports.get("risk", {})
    })


@app.get("/satisfaction", response_class=HTMLResponse)
async def satisfaction_page(request: Request):
    return templates.TemplateResponse(request=request, name="satisfaction.html", context={
        "active_page": "satisfaction",
        "satisfaction": cached_reports.get("satisfaction", {})
    })


@app.get("/explainability", response_class=HTMLResponse)
async def explainability_page(request: Request):
    return templates.TemplateResponse(request=request, name="explainability.html", context={
        "active_page": "explainability",
        "explainability": cached_reports.get("explainability", {})
    })


@app.get("/simulation", response_class=HTMLResponse)
async def simulation_page(request: Request):
    return templates.TemplateResponse(request=request, name="simulation.html", context={
        "active_page": "simulation",
        "simulation": cached_reports.get("simulation", {})
    })


@app.get("/audit", response_class=HTMLResponse)
async def audit_page(request: Request):
    return templates.TemplateResponse(request=request, name="audit.html", context={
        "active_page": "audit",
        "sample_orders": sample_orders[:25],
        "hypothesis": cached_reports.get("hypothesis", {})
    })


@app.get("/enterprise", response_class=HTMLResponse)
async def enterprise_page(request: Request):
    return templates.TemplateResponse(request=request, name="enterprise.html", context={
        "active_page": "enterprise"
    })


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return RedirectResponse(url="https://img.icons8.com/fluency/48/delivery-scooter.png")


@app.get("/viva", include_in_schema=False)
async def redirect_viva():
    return RedirectResponse(url="/enterprise", status_code=301)

# ------------------------------------------------------------------------------
# JSON REST API Endpoints
# ------------------------------------------------------------------------------
@app.post("/api/predict")
async def api_predict(order: OrderPredictionInput):
    try:
        result = run_prediction_pipeline(order.model_dump())
        return JSONResponse(content={"status": "success", "data": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulate")
async def api_simulate(sim_in: SimulationInput):
    sim_data = cached_reports.get("simulation", {})
    sensitivity = sim_data.get("threshold_sensitivity", [])
    if not sensitivity:
        raise HTTPException(status_code=404, detail="Simulation data not available")

    # Find closest threshold
    closest = min(sensitivity, key=lambda x: abs(x["threshold"] - sim_in.threshold))
    return JSONResponse(content={"status": "success", "data": closest})


@app.get("/api/metrics")
async def api_metrics():
    return JSONResponse(content={"status": "success", "reports": cached_reports})


@app.get("/api/health")
async def api_health():
    return JSONResponse(content={
        "status": "healthy",
        "service": "PromiseGuard Web Framework",
        "models_loaded": all([eta_pipe, q05_pipe, q95_pipe, risk_pipe]),
        "python_version": sys.version
    })
