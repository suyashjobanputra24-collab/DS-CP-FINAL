# PromiseGuard: Build Status and Progress Tracker

**Project ID**: MDM-DS-039  
**Course**: MD3135 Data Science | Third Year Computer Engineering (Academic Year 2026–27)  
**Project Title**: PromiseGuard: An Explainable Machine Learning Framework for Food Delivery ETA Reliability, Customer Dissatisfaction Prediction and Proactive Delay Prevention  
**Team**: Suyash Jobanputra & Shreyash Jobanputra  
**Guide**: Dr. Kapil N. Vhatkar  
**Last Updated**: 2026-11-19  

---

## 1. Executive Summary of Progress

| Phase | Description | Status | Completion % |
|---|---|---|---|
| **Phase 1** | Understand & Requirements Alignment | **COMPLETED** | 100% |
| **Phase 2** | Data Strategy, Acquisition & Governance Audit | **COMPLETED** | 100% |
| **Phase 3** | Data Cleaning, Pipeline & Feature Engineering | **COMPLETED** | 100% |
| **Phase 4** | Exploratory Data Analysis & Statistical Inference | **COMPLETED** | 100% |
| **Phase 5** | Machine Learning: ETA Regression & Baselines | **COMPLETED** | 100% |
| **Phase 6** | Prediction Reliability & Error Slice Analysis | **COMPLETED** | 100% |
| **Phase 7** | Uncertainty Estimation (Quantile & Conformal) | **COMPLETED** | 100% |
| **Phase 8** | Delivery Promise-Break Risk Modeling | **COMPLETED** | 100% |
| **Phase 9** | Customer Dissatisfaction Analytics | **COMPLETED** | 100% |
| **Phase 10** | Explainable Machine Learning (SHAP & Permutation) | **COMPLETED** | 100% |
| **Phase 11** | Proactive Intervention Simulation & Policy Trade-offs | **COMPLETED** | 100% |
| **Phase 12** | Interactive Multi-Page Web Platform (FastAPI + Jinja2) | **COMPLETED** | 100% |
| **Phase 13** | Streamlit Interactive Dashboard (app/app.py) | **COMPLETED** | 100% |
| **Phase 14** | Academic Documentation & Viva Defense Portal | **COMPLETED** | 100% |
| **Phase 15** | Automated Verification & Pipeline Testing (12/12 Tests) | **COMPLETED** | 100% |

---

## 2. Key Error Resolutions & Enhancements Completed

1. **Resolved Directory Nesting**: Flattened nested `data science project/data science project/` structure into the workspace root. All relative paths now resolve reliably from root.
2. **Fixed Broken Environment**: Removed invalid macOS `.venv` and verified system packages; installed `streamlit`, `plotly`, `shap`, `fastapi`, `uvicorn`, and `jinja2`.
3. **Eliminated Syntax Warnings**: Updated LaTeX escape strings (`\ge`, `\pm`, `\Delta`) across `src/reliability.py`, `src/uncertainty.py`, and `src/explainability.py` to prevent future Python syntax errors.
4. **Fixed Notebook Import Paths**: Regenerated all 11 Jupyter notebooks with directory-aware path bootstrapping, ensuring they execute cleanly whether run from the project root or inside `notebooks/`.
5. **Built Full-Stack Interactive Website**: Created a modern, multi-page web platform in `website/` (FastAPI backend + responsive HTML5/CSS3/JavaScript frontend) accessible at `http://127.0.0.1:8000` with live ML prediction, dynamic SLA timeline gauge, interactive policy simulator slider, error slicing, and REST API documentation at `/docs`.
6. **Built Automated Test Suite**: Expanded test coverage in `tests/test_website.py` and `tests/test_pipeline.py`. 12/12 unit and integration tests pass with 100% success.
