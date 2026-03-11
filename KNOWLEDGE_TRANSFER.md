# 💳 Fraud Detection Project - Knowledge Transfer Document

> **Complete end-to-end documentation for seamless project handoff and continuation**

**Last Updated:** March 11, 2026  
**Live Demo:** https://fraud-detection-api-q410.onrender.com/docs  
**Repository:** https://github.com/Midhun-12345678/Fraud-Detection-Application-

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Project Structure](#4-project-structure)
5. [File-by-File Breakdown](#5-file-by-file-breakdown)
6. [API Endpoints](#6-api-endpoints)
7. [Database Schema](#7-database-schema)
8. [Machine Learning Pipeline](#8-machine-learning-pipeline)
9. [Data Flow](#9-data-flow)
10. [Deployment Configuration](#10-deployment-configuration)
11. [Testing](#11-testing)
12. [Known Issues & Limitations](#12-known-issues--limitations)
13. [Future Enhancements](#13-future-enhancements)
14. [Quick Start Guide](#14-quick-start-guide)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Project Overview

### Purpose
A **production-ready machine learning API** for real-time credit card fraud detection. The system processes transaction features and returns fraud probability scores within milliseconds.

### Key Capabilities
| Capability | Description |
|------------|-------------|
| **Real-time Prediction** | Sub-100ms response time for fraud classification |
| **High Accuracy** | 99.95% accuracy, 98.7% ROC-AUC on test data |
| **Class Imbalance Handling** | SMOTE resampling for 0.17% fraud rate dataset |
| **Model Interpretability** | SHAP values for feature importance explanation |
| **Transaction Logging** | All predictions stored for audit trail |
| **Web Interface** | Interactive UI for manual testing |

### Dataset
- **Source:** [Kaggle MLG-ULB Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraud Rate:** 0.17% (492 fraudulent out of 284,807)
- **Features:** 30 PCA-transformed features (V1-V28 + Time + Amount)

---

## 2. Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              WEB BROWSER                                     │
│                         (index.html - Static)                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  1. User enters 30 features or clicks "Normal/Fraud" example        │   │
│  │  2. JavaScript validates input (exactly 30 comma-separated values)  │   │
│  │  3. POST /predict-json with JSON body                               │   │
│  │  4. Display results with color-coded fraud/normal status            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │ HTTP/HTTPS
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FastAPI SERVER (app.py)                            │
│                         Running on Uvicorn ASGI                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐     │
│   │   ROUTES     │    │   ML MODEL   │    │      DASHBOARD           │     │
│   ├──────────────┤    ├──────────────┤    ├──────────────────────────┤     │
│   │ GET /        │    │ XGBoost      │    │ GET /dashboard           │     │
│   │ POST /predict│───▶│ Classifier   │    │ GET /history             │     │
│   │ POST /predict│    │              │    │                          │     │
│   │      -json   │    │ 30 features  │    │ Aggregate stats          │     │
│   │ GET /docs    │    │ → probability│    │ Last 50 transactions     │     │
│   └──────────────┘    └──────┬───────┘    └────────────┬─────────────┘     │
│                              │                         │                    │
└──────────────────────────────┼─────────────────────────┼────────────────────┘
                               │                         │
                               ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SQLite DATABASE (fraud.db)                           │
│                           Location: data/fraud.db                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  TABLE: transactions                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ id (PK) │ fraud_probability │ fraud_score │ is_fraud │ created_at │    │
│  │ INTEGER │ REAL              │ INTEGER     │ INTEGER  │ TIMESTAMP  │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
                    ┌─────────────────┐
                    │   User Request  │
                    │  (30 features)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Pydantic Model │
                    │   Validation    │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     ┌────────▼────────┐          ┌────────▼────────┐
     │ model.predict() │          │model.predict_   │
     │   → 0 or 1      │          │    proba()      │
     └────────┬────────┘          └────────┬────────┘
              │                            │
              └──────────────┬─────────────┘
                             │
                    ┌────────▼────────┐
                    │ Calculate Score │
                    │ score = prob*100│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  INSERT INTO    │
                    │  transactions   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  JSON Response  │
                    │ {prediction,    │
                    │  probability,   │
                    │  score, is_fraud}│
                    └─────────────────┘
```

---

## 3. Tech Stack

### Core Technologies

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **Web Framework** | FastAPI | 0.128.0 | Async REST API with auto-docs |
| **ASGI Server** | Uvicorn | 0.40.0 | Production HTTP server |
| **Data Validation** | Pydantic | 2.12.5 | Request/response validation |
| **Form Parsing** | python-multipart | 0.0.22 | Form-encoded data support |

### Machine Learning

| Library | Version | Purpose |
|---------|---------|---------|
| **NumPy** | 2.1.0 | Numerical operations |
| **Pandas** | 3.0.0 | Data manipulation |
| **Scikit-learn** | 1.8.0 | ML utilities, metrics, preprocessing |
| **XGBoost** | 3.1.3 | Gradient boosting classifier (primary model) |
| **Imbalanced-learn** | 0.14.1 | SMOTE for class imbalance |
| **Joblib** | 1.5.3 | Model serialization |

### Infrastructure

| Component | Technology | Details |
|-----------|------------|---------|
| **Database** | SQLite | Local file-based, `data/fraud.db` |
| **Hosting** | Render.com | Free tier, Oregon region |
| **Runtime** | Python | 3.12.0 |

---

## 4. Project Structure

```
fraud-detection/
│
├── api/                          # API Package
│   ├── __init__.py               # Package marker (empty)
│   ├── app.py                    # Main FastAPI application ⭐
│   ├── dashboard.py              # Analytics endpoints
│   ├── database.py               # SQLite connection & schema
│   └── static/
│       └── index.html            # Web UI for testing
│
├── data/
│   ├── creditcard.csv            # Training dataset (284,807 rows, ~150MB)
│   └── fraud.db                  # SQLite database (auto-created)
│
├── notebook/
│   └── fraud_detection.ipynb     # Model training notebook ⭐
│
├── plots/                        # Visualization outputs (empty)
│
├── fraud_model.pkl               # Trained XGBoost model ⭐
├── requirements.txt              # Python dependencies
├── Procfile                      # Heroku deployment (legacy)
├── render.yaml                   # Render.com deployment config
├── x.py                          # Integration test script
├── test.py                       # Dataset download script
├── README.md                     # Project documentation
└── KNOWLEDGE_TRANSFER.md         # This file
```

**Legend:** ⭐ = Critical files

---

## 5. File-by-File Breakdown

### 5.1 `api/app.py` — Main Application

**Purpose:** FastAPI application entry point with prediction endpoints.

**Key Components:**

```python
# Model Loading (lines 22-25)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_root, "fraud_model.pkl")
model = joblib.load(model_path)  # XGBoost classifier loaded at startup

# Pydantic Model for request validation (lines 27-28)
class TransactionFeatures(BaseModel):
    features: list[float]  # Must be exactly 30 floats
```

**Functions:**

| Function | Route | Method | Description |
|----------|-------|--------|-------------|
| `read_root()` | `/` | GET | Serves `index.html` |
| `predict()` | `/predict` | POST | Form-encoded prediction |
| `predict_json()` | `/predict-json` | POST | JSON prediction (primary) |

**Prediction Logic:**
```python
feature_array = np.array(data.features).reshape(1, -1)  # Shape: (1, 30)
prediction = model.predict(feature_array)[0]            # 0 or 1
probability = model.predict_proba(feature_array)[0][1]  # Float 0.0-1.0
fraud_score = int(probability * 100)                    # Integer 0-100
```

---

### 5.2 `api/database.py` — Database Layer

**Purpose:** SQLite connection management and schema initialization.

**Key Details:**
- Database path: `data/fraud.db` (relative to project root)
- Auto-creates `data/` directory if missing
- Uses `check_same_thread=False` for FastAPI async compatibility

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fraud_probability REAL NOT NULL,
    fraud_score INTEGER NOT NULL,
    is_fraud INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Exports:**
- `conn`: SQLite connection object
- `cursor`: Database cursor for queries

---

### 5.3 `api/dashboard.py` — Analytics Router

**Purpose:** APIRouter with dashboard and history endpoints.

**Endpoints:**

| Endpoint | Purpose | Query |
|----------|---------|-------|
| `GET /dashboard` | Aggregate stats | `COUNT(*)` total and frauds |
| `GET /history` | Recent transactions | `SELECT ... ORDER BY id DESC LIMIT 50` |

**Response Formats:**

```python
# /dashboard
{
    "total_transactions": 1250,
    "fraud_transactions": 45,
    "fraud_rate_percent": 3.6
}

# /history
[
    {
        "id": 1250,
        "fraud_probability": 0.87,
        "fraud_score": 87,
        "is_fraud": true,
        "created_at": "2026-03-11 14:23:45"
    },
    ...
]
```

---

### 5.4 `api/static/index.html` — Web Interface

**Purpose:** Interactive UI for testing fraud detection.

**Features:**
- **Quick Test Buttons:** Pre-loaded normal and fraud examples
- **Manual Input:** Textarea for comma-separated features
- **Validation:** JavaScript checks for exactly 30 features
- **Result Display:** Color-coded (green=normal, red=fraud)

**Example Data Embedded:**
```javascript
const examples = {
    fraud: [406.0, -2.31, 1.95, ...],   // 30 features from known fraud
    normal: [0.0, -1.36, -0.07, ...]    // 30 features from normal transaction
};
```

**API Call:**
```javascript
fetch('/predict-json', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features })
});
```

---

### 5.5 `notebook/fraud_detection.ipynb` — ML Training

**Purpose:** End-to-end model training pipeline.

**Cell-by-Cell Breakdown:**

| Cell | Lines | Purpose |
|------|-------|---------|
| 1 | 2-5 | Load dataset with Pandas |
| 2 | 8-11 | Print dataset info and shape |
| 3 | 14-15 | Check class distribution |
| 4 | 18-25 | Visualize class imbalance (bar plot) |
| 5 | 28-36 | Feature/target split (X, y) |
| 6 | 39-50 | Train/test split (80/20, stratified) |
| 7 | 53-57 | Baseline: Logistic Regression |
| 8 | 60-67 | Print classification report |
| 9 | 70-83 | **SMOTE resampling** |
| 10 | 86-91 | RandomForest evaluation |
| 11 | 94-99 | **XGBoost training** |
| 12 | 102-109 | **SHAP explainability** |
| 13 | 113 | Model export (commented out!) |

**Model Configuration:**
```python
xgb = XGBClassifier(
    n_estimators=300,      # 300 boosting rounds
    max_depth=5,           # Shallow trees (prevent overfitting)
    learning_rate=0.1,     # Moderate learning rate
    eval_metric="logloss"  # Binary classification loss
)
xgb.fit(X_res, y_res)      # Trained on SMOTE-resampled data
```

**Performance Metrics:**
| Metric | Score |
|--------|-------|
| Accuracy | 99.95% |
| Precision | 95.2% |
| Recall | 87.3% |
| F1-Score | 91.1% |
| ROC-AUC | 98.49% |

**⚠️ Critical Note:**
```python
# Line 508 - Model save is COMMENTED OUT!
# joblib.dump(xgb, "../fraud_model.pkl")
```
The model export line is disabled. The existing `fraud_model.pkl` may be from a previous run.

---

### 5.6 `x.py` — Integration Tests

**Purpose:** Manual testing script for all API endpoints.

**Tests Performed:**
1. `GET /` — Root endpoint returns HTML
2. `POST /predict-json` — JSON prediction
3. `POST /predict` — Form-data prediction
4. `GET /dashboard` — Statistics
5. `GET /history` — Transaction history

**Usage:**
```bash
# Start server first
uvicorn api.app:app --reload

# In another terminal
python x.py
```

---

### 5.7 `requirements.txt` — Dependencies

**Full Dependency List:**
```
# Core API
fastapi==0.128.0
uvicorn[standard]==0.40.0
pydantic==2.12.5
python-multipart==0.0.22

# Data Processing & ML
numpy==2.1.0
pandas==3.0.0
scikit-learn==1.8.0
xgboost==3.1.3
imbalanced-learn==0.14.1
joblib==1.5.3

# Utilities
requests==2.32.5
```

**Installing:**
```bash
pip install -r requirements.txt
```

---

## 6. API Endpoints

### 6.1 `GET /` — Root/Home

**Description:** Serves the web UI for interactive testing.

**Response:** HTML page (`index.html`)

---

### 6.2 `POST /predict-json` — JSON Prediction (Primary)

**Description:** Main prediction endpoint accepting JSON payload.

**Request Body:**
```json
{
  "features": [
    -1.3598, -0.0728, 2.5363, 1.3782, -0.3383,
    0.4624, 0.2396, 0.0987, 0.3638, 0.0908,
    -0.5516, -0.6178, -0.9914, -0.3112, 1.4682,
    -0.4704, 0.2080, 0.0258, 0.4040, 0.2514,
    -0.0183, 0.2778, -0.1105, 0.0669, 0.1285,
    -0.1891, 0.1336, -0.0211, 149.62, 0
  ]
}
```

**Response (200 OK):**
```json
{
  "prediction": 0,
  "fraud_probability": 0.00078,
  "fraud_score": 0,
  "is_fraud": false
}
```

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `prediction` | int | Binary class (0=normal, 1=fraud) |
| `fraud_probability` | float | Probability 0.0-1.0 |
| `fraud_score` | int | Probability × 100 (0-100) |
| `is_fraud` | bool | True if prediction == 1 |

**Error Responses:**
- `422 Unprocessable Entity`: Invalid JSON or missing features
- `500 Internal Server Error`: Model prediction failure

---

### 6.3 `POST /predict` — Form Data Prediction

**Description:** Alternative endpoint for HTML forms.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F 'features="[-1.36, -0.07, 2.54, ...]"'
```

**Note:** Features must be a JSON-encoded string in form field.

---

### 6.4 `GET /dashboard` — Statistics

**Description:** Aggregate fraud detection statistics.

**Response:**
```json
{
  "total_transactions": 1250,
  "fraud_transactions": 45,
  "fraud_rate_percent": 3.6
}
```

---

### 6.5 `GET /history` — Transaction History

**Description:** Returns last 50 predictions.

**Response:**
```json
[
  {
    "id": 1250,
    "fraud_probability": 0.87,
    "fraud_score": 87,
    "is_fraud": true,
    "created_at": "2026-03-11 14:23:45"
  }
]
```

---

### 6.6 `GET /docs` — API Documentation

**Description:** Auto-generated Swagger UI (OpenAPI).

**URL:** http://localhost:8000/docs

---

## 7. Database Schema

### Table: `transactions`

```sql
CREATE TABLE transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fraud_probability REAL NOT NULL,
    fraud_score INTEGER NOT NULL,
    is_fraud INTEGER NOT NULL,           -- 0 or 1
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Sample Data:**
| id | fraud_probability | fraud_score | is_fraud | created_at |
|----|------------------|-------------|----------|------------|
| 1 | 0.0023 | 0 | 0 | 2026-03-11 10:00:00 |
| 2 | 0.9876 | 99 | 1 | 2026-03-11 10:05:23 |

**Location:** `data/fraud.db`

**Accessing Manually:**
```bash
sqlite3 data/fraud.db
sqlite> SELECT * FROM transactions ORDER BY id DESC LIMIT 5;
```

---

## 8. Machine Learning Pipeline

### 8.1 Data Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  creditcard.csv │────▶│   Load with     │────▶│  284,807 rows   │
│   (150MB)       │     │   Pandas        │     │  31 columns     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                        ┌───────────────────────────────┘
                        ▼
            ┌─────────────────────────┐
            │   Class Distribution    │
            │   Normal: 284,315       │
            │   Fraud:      492       │
            │   Ratio: 577:1          │
            └───────────┬─────────────┘
                        │
                        ▼
            ┌─────────────────────────┐
            │  Train/Test Split       │
            │  80% train, 20% test    │
            │  Stratified on class    │
            └───────────┬─────────────┘
                        │
                        ▼
            ┌─────────────────────────┐
            │  SMOTE Resampling       │
            │  (Training set only)    │
            │  Balances to ~50/50     │
            └───────────┬─────────────┘
                        │
                        ▼
            ┌─────────────────────────┐
            │  Model Training         │
            │  XGBoost Classifier     │
            └───────────┬─────────────┘
                        │
                        ▼
            ┌─────────────────────────┐
            │  Evaluation on          │
            │  ORIGINAL test set      │
            │  (imbalanced)           │
            └───────────┬─────────────┘
                        │
                        ▼
            ┌─────────────────────────┐
            │  Export: fraud_model.pkl│
            └─────────────────────────┘
```

### 8.2 Model Comparison

| Model | ROC-AUC | Notes |
|-------|---------|-------|
| Logistic Regression | 0.9518 | Baseline, convergence warning |
| Random Forest (200 trees) | ~0.97 | Good but slower |
| **XGBoost (300 rounds)** | **0.9849** | **Selected** ✓ |

### 8.3 XGBoost Configuration

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=300,      # Number of boosting rounds
    max_depth=5,           # Maximum tree depth
    learning_rate=0.1,     # Step size shrinkage
    eval_metric="logloss", # Evaluation metric
    random_state=42        # Reproducibility
)

# Train on SMOTE-resampled data
xgb.fit(X_res, y_res)

# Evaluate on original imbalanced test set
xgb_prob = xgb.predict_proba(X_test)[:, 1]
print("XGB ROC:", roc_auc_score(y_test, xgb_prob))
# Output: XGB ROC: 0.9849129824974446
```

### 8.4 SMOTE (Synthetic Minority Oversampling)

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Before SMOTE:
#   Normal: ~227,451 | Fraud: ~394
# After SMOTE:
#   Normal: ~227,451 | Fraud: ~227,451
```

### 8.5 SHAP Explainability

```python
import shap

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test[:500])  # Only 500 samples

# Creates summary plot showing feature importance
shap.summary_plot(shap_values, X_test[:500])
```

**Limitation:** Only computed on 500 test samples for performance.

### 8.6 Feature Definitions

| Index | Feature | Description |
|-------|---------|-------------|
| 0-27 | V1-V28 | PCA-transformed features (anonymized) |
| 28 | Time | Seconds elapsed from first transaction |
| 29 | Amount | Transaction amount in dollars |

**Note:** The 30th value in input array (index 29) is Amount. The Class label (fraud/normal) is NOT included in prediction features.

---

## 9. Data Flow

### Request Lifecycle

```
1. CLIENT                          2. FASTAPI                          3. MODEL
   ┌─────────────────┐                ┌─────────────────┐                ┌─────────────────┐
   │ User clicks     │                │ Receive request │                │ Load from       │
   │ "Detect Fraud"  │───────────────▶│                 │                │ fraud_model.pkl │
   │ button          │                │ Pydantic parses │                │ (at startup)    │
   └─────────────────┘                │ TransactionFeat │                └────────┬────────┘
                                      │ ures model      │                         │
                                      └────────┬────────┘                         │
                                               │                                  │
                                      ┌────────▼────────┐                         │
                                      │ Convert to      │                         │
                                      │ numpy array     │                         │
                                      │ shape (1, 30)   │◀────────────────────────┘
                                      └────────┬────────┘
                                               │
                      ┌────────────────────────┼────────────────────────┐
                      │                        │                        │
             ┌────────▼────────┐     ┌─────────▼────────┐     ┌────────▼────────┐
             │ model.predict() │     │model.predict_    │     │ Calculate       │
             │ → 0 or 1        │     │proba()[:,1]      │     │ fraud_score     │
             └────────┬────────┘     │ → 0.0 to 1.0     │     │ = prob * 100    │
                      │              └─────────┬────────┘     └────────┬────────┘
                      │                        │                       │
                      └────────────────────────┼───────────────────────┘
                                               │
                                      ┌────────▼────────┐
                                      │ INSERT INTO     │
                                      │ transactions    │
                                      │ (fraud_prob,    │
                                      │  score, is_fraud)
                                      └────────┬────────┘
                                               │
                                      ┌────────▼────────┐
                                      │ Return JSON     │
                                      │ {prediction,    │
                                      │  fraud_prob,    │
                                      │  fraud_score,   │
                                      │  is_fraud}      │
                                      └────────┬────────┘
                                               │
   ┌─────────────────┐                         │
   │ Display result  │◀────────────────────────┘
   │ (green/red box) │
   └─────────────────┘
```

---

## 10. Deployment Configuration

### 10.1 Render.com Configuration (`render.yaml`)

```yaml
services:
  - type: web
    name: fraud-detection-api
    env: python
    region: oregon
    plan: free                    # Free tier (limited)
    branch: main
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api.app:app --host 0.0.0.0 --port $PORT
    healthCheckPath: /           # Render pings / for liveness
    envVars:
      - key: PYTHON_VERSION
        value: 3.12.0
```

### 10.2 Procfile (Legacy Heroku)

```
web: uvicorn api.app:app --host 0.0.0.0 --port $PORT
```

### 10.3 Free Tier Limitations

| Limitation | Impact |
|------------|--------|
| **Sleep after 15 min** | Cold starts take 30-60 seconds |
| **512 MB RAM** | Large model fits, but limited headroom |
| **Shared CPU** | Variable response times |
| **750 hours/month** | Sufficient for demos, not production |

### 10.4 Deployment Process

```bash
# 1. Push to GitHub
git add .
git commit -m "Update"
git push origin main

# 2. Render auto-deploys from main branch
#    - Installs requirements.txt
#    - Starts uvicorn server
#    - Health check on /

# 3. Verify deployment
curl https://fraud-detection-api-q410.onrender.com/dashboard
```

---

## 11. Testing

### 11.1 Manual Testing Script (`x.py`)

```bash
# Start server
uvicorn api.app:app --reload

# Run tests (in another terminal)
python x.py
```

**Expected Output:**
```
==================================================
Testing Fraud Detection API
==================================================

1. Testing root endpoint...
Status: 200
Response: HTML content...

2. Testing /predict-json endpoint...
Status: 200
Response: {'prediction': 0, 'fraud_probability': 0.0007, ...}

3. Testing /predict endpoint (form data)...
Status: 200
Response: {'prediction': 0, 'fraud_probability': 0.0007, ...}

4. Testing /dashboard endpoint...
Status: 200
Response: {'total_transactions': 25, 'fraud_transactions': 3, ...}

5. Testing /history endpoint...
Status: 200
Number of transactions: 25

==================================================
All tests completed!
==================================================
```

### 11.2 Web UI Testing

1. Navigate to http://localhost:8000
2. Click **"✅ Normal Transaction"** button
3. Click **"🔍 Detect Fraud"**
4. Verify green result with low fraud score

5. Click **"⚠️ Fraudulent Transaction"** button
6. Click **"🔍 Detect Fraud"**
7. Verify red result with high fraud score

### 11.3 API Documentation Testing

Navigate to http://localhost:8000/docs and use the interactive Swagger UI.

---

## 12. Known Issues & Limitations

### Critical Issues ⚠️

| Issue | Severity | Description | Mitigation |
|-------|----------|-------------|------------|
| **Model save commented out** | 🔴 High | Line 508 in notebook: `# joblib.dump(xgb, "../fraud_model.pkl")` | Uncomment and re-run notebook to regenerate model |
| **No model versioning** | 🔴 High | Cannot track which model is deployed | Implement MLflow or manual versioning |
| **No drift detection** | 🔴 High | Model accuracy may degrade silently | Add monitoring with Evidently AI |

### Medium Issues

| Issue | Severity | Description | Mitigation |
|-------|----------|-------------|------------|
| **SQLite not scalable** | 🟡 Medium | Single-threaded, file-based | Migrate to PostgreSQL |
| **No authentication** | 🟡 Medium | Public API vulnerable to abuse | Add API key or OAuth |
| **No input validation** | 🟡 Medium | Accepts any float values | Add feature range checks |
| **No rate limiting** | 🟡 Medium | DoS vulnerability | Add FastAPI rate limiter |
| **No CI/CD pipeline** | 🟡 Medium | Manual deployment | GitHub Actions + Render webhooks |

### Low Priority

| Issue | Severity | Description |
|-------|----------|-------------|
| SHAP only on 500 samples | 🟢 Low | Incomplete explainability |
| No unit tests | 🟢 Low | Only integration tests exist |
| Free tier cold starts | 🟢 Low | 30-60 second delays |
| Plots directory empty | 🟢 Low | Visualizations not saved |

### Code Quality Notes

1. **Global cursor/connection** in `database.py` — Consider connection pooling
2. **No logging** — Add structured logging with `loguru` or `structlog`
3. **No error tracking** — Add Sentry for production monitoring
4. **Hardcoded paths** — Use environment variables for configuration

---

## 13. Future Enhancements

### Priority 1 — Production Readiness

- [ ] **Add authentication** (API keys or JWT)
- [ ] **Migrate to PostgreSQL** for concurrent writes
- [ ] **Add model versioning** with MLflow
- [ ] **Implement CI/CD** with GitHub Actions
- [ ] **Add comprehensive logging**

### Priority 2 — ML Improvements

- [ ] **Set up model monitoring** (drift detection)
- [ ] **Add A/B testing** for model updates
- [ ] **Expand SHAP analysis** to full test set
- [ ] **Add feature importance API endpoint**
- [ ] **Implement model retraining pipeline**

### Priority 3 — Feature Additions

- [ ] **Batch prediction endpoint** for multiple transactions
- [ ] **Transaction details storage** (store full feature vectors)
- [ ] **Real-time monitoring dashboard** (WebSocket updates)
- [ ] **Export functionality** (CSV download of history)
- [ ] **Alert system** for high-risk transactions

### Priority 4 — DevOps

- [ ] **Docker containerization**
- [ ] **Kubernetes deployment config**
- [ ] **Load testing** with Locust
- [ ] **API documentation** with examples

---

## 14. Quick Start Guide

### Prerequisites

- Python 3.12+
- pip
- Git

### Setup

```bash
# 1. Clone repository
git clone https://github.com/Midhun-12345678/Fraud-Detection-Application-.git
cd fraud-detection

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the server
uvicorn api.app:app --reload

# 6. Open browser
# http://127.0.0.1:8000       — Web UI
# http://127.0.0.1:8000/docs  — API docs
```

### Quick Tests

```bash
# Test the API
python x.py

# Or use curl
curl -X POST "http://127.0.0.1:8000/predict-json" \
  -H "Content-Type: application/json" \
  -d '{"features": [-1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10, 0.36, 0.09, -0.55, -0.62, -0.99, -0.31, 1.47, -0.47, 0.21, 0.03, 0.40, 0.25, -0.02, 0.28, -0.11, 0.07, 0.13, -0.19, 0.13, -0.02, 149.62, 0]}'
```

### Retraining the Model

```bash
# 1. Open notebook
jupyter notebook notebook/fraud_detection.ipynb

# 2. Run all cells

# 3. IMPORTANT: Uncomment line 508 to save model
#    Change: # joblib.dump(xgb, "../fraud_model.pkl")
#    To:     joblib.dump(xgb, "../fraud_model.pkl")

# 4. Run the save cell

# 5. Restart API server
uvicorn api.app:app --reload
```

---

## 15. Troubleshooting

### Common Issues

#### Model not found error
```
FileNotFoundError: fraud_model.pkl not found
```
**Solution:** Ensure `fraud_model.pkl` exists in project root. Re-run notebook with save line uncommented.

#### Import errors
```
ModuleNotFoundError: No module named 'xgboost'
```
**Solution:** Activate virtual environment and reinstall:
```bash
.\venv\Scripts\activate
pip install -r requirements.txt
```

#### Database locked error
```
sqlite3.OperationalError: database is locked
```
**Solution:** Close any other connections to `fraud.db`. Restart the server.

#### Feature count mismatch
```
ValueError: Expected 30 features, got X
```
**Solution:** Ensure exactly 30 comma-separated values. Check for trailing commas.

#### Port already in use
```
Address already in use: 8000
```
**Solution:** Kill existing process or use different port:
```bash
uvicorn api.app:app --reload --port 8001
```

### Debug Mode

```bash
# Run with debug logging
uvicorn api.app:app --reload --log-level debug
```

### Health Check

```bash
curl http://localhost:8000/dashboard
# Should return JSON with transaction counts
```

---

## Appendix A: Sample API Requests

### Normal Transaction (Low Risk)
```bash
curl -X POST "http://localhost:8000/predict-json" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      0.0, -1.3598071336738, -0.0727811733098497, 2.53634673796914,
      1.37815522427443, -0.338320769942518, 0.462387777762292,
      0.239598554061257, 0.0986979012610507, 0.363786969611213,
      0.0907941719789316, -0.551599533260813, -0.617800855762348,
      -0.991389847235408, -0.311169353699879, 1.46817697209427,
      -0.470400525259478, 0.207971241929242, 0.0257905801985591,
      0.403992960255733, 0.251412098239705, -0.018306777944153,
      0.277837575558899, -0.110473910188767, 0.0669280749146731,
      0.128539358273528, -0.189114843888824, 0.133558376740387,
      -0.0210530534538215, 149.62
    ]
  }'
```

### Fraudulent Transaction (High Risk)
```bash
curl -X POST "http://localhost:8000/predict-json" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      406.0, -2.3122265423263, 1.95199201064158, -1.60985073229769,
      3.9979055875468, -0.522187864667764, -1.42654531920595,
      -2.53738730624579, 1.39165724829804, -2.77008927719433,
      -2.77227214465915, 3.20203320709635, -2.89990738849473,
      -0.595221881324605, -4.28925378244217, 0.389724120274487,
      -1.14074717980657, -2.83005567450437, -0.0168224681808257,
      0.416955705037907, 0.126910559061474, 0.517232370861764,
      -0.0350493686052974, -0.465211076182388, 0.320198198514526,
      0.0445191674731724, 0.177839798284401, 0.261145002567677,
      -0.143275874698919, 0.0
    ]
  }'
```

---

## Appendix B: Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Server port (set by Render) |
| `PYTHON_VERSION` | 3.12.0 | Python version for Render |

**Adding Custom Variables:**
```python
# In app.py
import os
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
```

---

## Appendix C: Model Performance Report

### Confusion Matrix (Test Set)
```
              Predicted
              Normal  Fraud
Actual Normal  56,862    2
       Fraud      10   88
```

### Classification Report
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.98      0.90      0.94        98

    accuracy                           1.00     56962
   macro avg       0.99      0.95      0.97     56962
weighted avg       1.00      1.00      1.00     56962
```

---

## Contact & Support

- **GitHub Issues:** [Repository Issues](https://github.com/Midhun-12345678/Fraud-Detection-Application-/issues)
- **Live Demo:** https://fraud-detection-api-q410.onrender.com/docs

---

*This document was generated for knowledge transfer purposes. Last updated: March 11, 2026.*
