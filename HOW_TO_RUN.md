# 🚀 PromiseGuard™ Enterprise — Quickstart Guide for VS Code

This guide walks you through setting up and running **PromiseGuard™ Enterprise** in Visual Studio Code.

---

## ⚡ 1-Minute Quick Start (Terminal)

Open VS Code, press **`Ctrl + \``** (or `Cmd + \`` on macOS) to open the integrated terminal, and run:

### Step 1: Create & Activate Virtual Environment (Recommended)

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 2: Install Required Packages

```bash
pip install -r requirements.txt
```

---

### Step 3: Run the Commercial Web Platform

```bash
python run_website.py
```
* **URL:** [http://127.0.0.1:8000](http://127.0.0.1:8000) (opens automatically in your browser)
* **Interactive API Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

*(On Windows, you can also simply double-click **`start_website.bat`**).*

---

### Step 4 (Optional): Run the Streamlit Analytics Console

If you want to explore the multi-page diagnostic and machine learning experimentation dashboard:
```bash
python -m streamlit run app/app.py
```
* **URL:** [http://localhost:8501](http://localhost:8501)

---

## 🧪 Run Automated Verification Tests

To verify that all models, data pipelines, web routes, and REST APIs are working properly:
```bash
python -m pytest tests/ -v
```
All **12 tests** should pass green in ~3 seconds.

---

## 💻 VS Code Configuration (No Missing Import Warnings)

This repository includes a pre-configured [`.vscode/settings.json`](.vscode/settings.json):
1. Press **`Ctrl + Shift + P`** (or `Cmd + Shift + P` on Mac).
2. Type **`Python: Select Interpreter`** and hit Enter.
3. Select your created virtual environment:
   * **`Python (.venv): .venv\Scripts\python.exe`** (or your main Python installation).
4. VS Code will immediately resolve all imports without warnings.

---

## 📁 Key Project Modules & Routes

* [`website/server.py`](website/server.py): FastAPI web server and REST API engine.
* [`website/templates/`](website/templates/): High-contrast obsidian dark UI templates.
* [`website/static/`](website/static/): Custom design system styles and reactive client scripts.
* [`app/app.py`](app/app.py): Streamlit diagnostic analytics dashboard.
* [`models/`](models/): Serialized machine learning pipelines (`joblib`).
* [`data/processed/`](data/processed/): Cleaned metropolitan order logs and test datasets.
* [`tests/`](tests/): Complete automated test suite.
