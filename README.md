#  Real-Time Fraud Detection System

> **Production ML system that scores transactions in <100ms, monitors for data drift, and auto-retrains when model performance degrades**

[![Live API](https://img.shields.io/badge/API-Live%20on%20Render-brightgreen)](https://fraud-detection-api-q410.onrender.com/docs)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.1-orange.svg)](https://xgboost.readthedocs.io/)

---

##  Business Problem

- **Fraud costs financial institutions 5-7% of annual revenue**  a $10B bank loses $500-700M yearly
- **Manual review doesn't scale**  human analysts can't process thousands of transactions per second
- **Black-box models create compliance risk**  regulators demand explainability for declined transactions
- **This system solves all three**  real-time scoring with explainable AI, so analysts see exactly *WHY* a transaction was flagged

---

##  Results

Trained on 284,807 real credit card transactions from [Kaggle MLG-ULB](https://www.kaggle.com/mlg-ulb/creditcardfraud):

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.95% |
| **ROC-AUC** | 98.70% |
| **Precision (Fraud)** | 98% |
| **Recall (Fraud)** | 90% |
| **Inference Time** | <100ms |
| **False Positives** | 2 out of 56,864 legitimate transactions |

**Confusion Matrix (Test Set):**
```
                 Predicted Normal    Predicted Fraud
Actual Normal         56,862              2
Actual Fraud             10             88
```

---

##  System Architecture

```

                              USER / CLIENT                                   

                                      
                                      

                         FastAPI PREDICTION SERVICE                           
                   
    /predict-json    XGBoost Model    SHAP Explainer          
     (<100ms)            (fraud_model)        (Top 5 reasons)         
                   

                                                         
                                                         
                    
   SQLite Audit Log                            STREAMLIT DASHBOARD        
  (Every prediction)                        Live Monitor   SHAP Explorer
                       Drift Monitor  Auto Retrain 
                                           
                                                          
                                                          

                         DRIFT DETECTION (Evidently)                          
         Compares live traffic distribution vs training data                  
                    Alerts when drift > 30% threshold                         

                                      
                                       (if drift detected)

                    AUTO-RETRAIN PIPELINE (XGBoost + SMOTE)                   
           
   Train New     Evaluate     Champion/Challenger Compare     
     Model            on F1           Promote ONLY if new model wins  
           

                                      
                                      

                    HOT-RELOAD TO PRODUCTION (Zero Downtime)                  

```

---

##  Key Features

### 1. Real-Time Scoring API
**Business value:** Every transaction gets a fraud probability score in under 100ms  fast enough for checkout flows.
- FastAPI async endpoints handle concurrent requests
- Returns probability (0-1), fraud score (0-100), and binary classification
- Every prediction logged for audit compliance

### 2. Explainable AI (SHAP)
**Business value:** Regulators and analysts can audit every decision  no more "black box" model risk.
- Top 5 features driving each prediction returned in API response
- Visual SHAP waterfall charts in Streamlit dashboard
- Answers: "Why was this $500 transaction flagged but not the $5000 one?"

### 3. Data Drift Detection (Evidently)
**Business value:** Know immediately when fraud patterns change  before your model goes stale.
- Compares live transaction distributions against training data
- Alerts when >30% of features show statistical drift
- Historical drift scores tracked over time

### 4. Auto-Retraining Pipeline
**Business value:** Model stays accurate without manual intervention  but never worse than current production.
- Triggered when drift detected OR accuracy drops below 94%
- New model trains on full dataset with SMOTE for class imbalance
- **Champion/Challenger:** New model only promoted if it beats current model on F1 score
- Production model hot-reloads with zero downtime

---

##  Tech Stack

| Layer | Technology | Why This Choice |
|-------|------------|-----------------|
| **API Framework** | FastAPI | Async, auto-generated docs, production-ready |
| **ML Model** | XGBoost | Best-in-class for tabular fraud detection |
| **Explainability** | SHAP | Industry standard, regulator-friendly |
| **Drift Detection** | Evidently | Purpose-built for ML monitoring |
| **Database** | SQLite | Lightweight audit trail, zero config |
| **Class Imbalance** | SMOTE | Handles extreme 577:1 class ratio |
| **Dashboard** | Streamlit + Plotly | Rapid ML dashboard development |
| **Deployment** | Render.com | Free tier, always-on, auto-deploy from GitHub |

---

##  Live Demo

** API Documentation (Swagger):**  
https://fraud-detection-api-q410.onrender.com/docs

** Quick Test  Fraud Example:**
```bash
curl -X POST "https://fraud-detection-api-q410.onrender.com/predict-json" \
  -H "Content-Type: application/json" \
  -d '{"features": [406.0,-2.312,1.952,-1.610,3.998,-0.522,-1.427,-2.537,1.392,-2.770,-2.772,3.202,-2.900,-0.595,-4.289,0.390,-1.141,-2.830,-0.017,0.417,0.127,0.517,-0.035,-0.465,0.320,0.045,0.178,0.261,-0.143,0.0]}'
```

** Quick Test  Normal Example:**
```bash
curl -X POST "https://fraud-detection-api-q410.onrender.com/predict-json" \
  -H "Content-Type: application/json" \
  -d '{"features": [0.0,-1.360,-0.073,2.536,1.378,-0.338,0.462,0.240,0.099,0.364,0.091,-0.552,-0.618,-0.991,-0.311,1.468,-0.470,0.208,0.026,0.404,0.251,-0.018,0.278,-0.110,0.067,0.129,-0.189,0.134,-0.021,149.62]}'
```

---

##  Project Structure

```
fraud-detection/
 api/
    app.py              # FastAPI app, model loading, /predict-json endpoint
    dashboard.py        # /dashboard, /history, /drift, /retrain endpoints
    database.py         # SQLite connection, transactions table
    static/index.html   # Simple web UI for manual testing
 drift/
    drift_detector.py   # Evidently-based drift detection logic
 retrain/
    retrain_pipeline.py # Auto-retraining with champion/challenger
 data/
    creditcard.csv      # Training data (284,807 transactions)
    fraud.db            # SQLite database for prediction logs
    drift_log.json      # Historical drift check results
    retrain_log.json    # Historical retraining results
 notebook/
    fraud_detection.ipynb # EDA, model training, evaluation
 streamlit_app.py        # 6-tab monitoring dashboard
 fraud_model.pkl         # Production XGBoost model
 requirements.txt        # Python dependencies
 Procfile                # Render.com deployment config
 README.md               # You are here
```

---

##  Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/Midhun-12345678/Fraud-Detection-Application-.git
cd Fraud-Detection-Application-

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start FastAPI server
uvicorn api.app:app --reload --port 8000

# 5. Start Streamlit dashboard (new terminal)
streamlit run streamlit_app.py --server.port 8501

# 6. Open in browser
# API Docs: http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

---

##  API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict-json` | POST | Score a transaction (returns probability, SHAP values, top 5 features) |
| `/dashboard` | GET | Get aggregate stats: total transactions, fraud count, fraud rate |
| `/history` | GET | Last 50 predictions with scores and timestamps |
| `/drift` | GET | Run drift check against training data (takes ~10s) |
| `/drift/history` | GET | Last 20 drift check results |
| `/retrain` | POST | Trigger retraining pipeline (takes 2-5 min) |
| `/retrain/current-metrics` | GET | Current model accuracy, precision, recall, F1, ROC-AUC |
| `/retrain/status` | GET | Last 10 retraining runs with outcomes |

---

##  MLOps Pipeline

**Automated model lifecycle in plain English:**

1. **Score & Log**  Every transaction gets a fraud probability; result is logged to SQLite for audit
2. **Monitor Drift**  Evidently compares live transaction patterns vs training data distribution
3. **Trigger Conditions**  Retraining starts if:
   - Drift score exceeds 30% threshold, OR
   - Model accuracy drops below 94%
4. **Train New Model**  Full dataset (284K rows) + SMOTE to handle class imbalance
5. **Champion/Challenger**  New model evaluated on held-out test set:
   - If new F1 > current F1  **Promoted**
   - If new F1  current F1  **Rejected** (current model preserved)
6. **Hot Reload**  Promoted model replaces production model with zero downtime

---

##  Business Impact Calculation

**Assumptions:**
- Average fraud transaction value: **$847**
- Model recall (fraud catch rate): **90%**
- False positive review cost: **$15/case**
- Daily transaction volume: **10,000**
- Fraud rate: **0.17%** (matches training data)

**Daily Impact:**
```
Fraud transactions/day:     10,000  0.17% = 17
Fraud caught (90% recall):  17  90% = 15.3
Fraud prevented value:      15.3  $847 = $12,959

False positives (from test): 2 per 56,864 = 0.0035%
FP per 10K transactions:    10,000  0.0035% = 0.35/day
FP cost:                    0.35  $15 = $5.25

Net daily savings:          $12,959 - $5.25 = $12,954
```

**Annual Impact (per 10,000 daily transactions):**
| Metric | Value |
|--------|-------|
| Fraud prevented | **$4.7M** |
| False positive cost | **$1,916** |
| **Net annual savings** | **$4.7M** |

---

##  Future Enhancements

- [ ] **Real-time WebSocket updates**  Push fraud alerts to dashboards instantly
- [ ] **Batch prediction endpoint**  Score CSV uploads with thousands of transactions
- [ ] **Kubernetes deployment**  Horizontal scaling for high-traffic production
- [ ] **A/B testing framework**  Gradually roll out new models to % of traffic
- [ ] **Graph neural network**  Detect fraud rings by analyzing transaction networks
- [ ] **Feature store integration**  Centralized feature management with Feast

---

##  Author

**Midhun**  
[GitHub](https://github.com/Midhun-12345678) 

---

##  License

This project is for educational and portfolio purposes. The dataset is provided by [Kaggle MLG-ULB](https://www.kaggle.com/mlg-ulb/creditcardfraud) under their terms of use.
