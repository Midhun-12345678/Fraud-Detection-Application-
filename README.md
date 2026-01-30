# 💳 Fraud Detection API

A production-ready machine learning API for real-time credit card fraud detection using XGBoost and FastAPI.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128.0-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

🚀 **Live Demo:** [https://fraud-detection-api-q410.onrender.com/docs](https://fraud-detection-api-q410.onrender.com/docs)
---

## 📊 Overview

This project implements a complete fraud detection pipeline with:
- **Machine Learning Model**: XGBoost trained on imbalanced credit card transaction data
- **REST API**: FastAPI backend with multiple prediction endpoints
- **Database**: SQLite for transaction logging and analytics
- **Dashboard**: Real-time fraud statistics and transaction history

### Key Features

✅ Real-time fraud prediction with 99%+ accuracy  
✅ RESTful API with JSON and form-data support  
✅ Transaction history tracking  
✅ Interactive API documentation (Swagger UI)  
✅ SMOTE-based class imbalance handling  
✅ Model interpretability with SHAP values

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | FastAPI, Uvicorn |
| **ML Framework** | Scikit-learn, XGBoost, Imbalanced-learn |
| **Data Processing** | Pandas, NumPy |
| **Database** | SQLite |
| **Deployment** | Render |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- pip

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the API**
```bash
uvicorn api.app:app --reload
```

The API will be available at `http://127.0.0.1:8000`

---

## 📡 API Endpoints

### Prediction Endpoints

#### POST `/predict-json`
Predict fraud probability using JSON payload

**Request:**
```json
{
  "features": [
    -1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10,
    0.36, 0.09, -0.55, -0.62, -0.99, -0.31, 1.47, -0.47,
    0.21, 0.03, 0.40, 0.25, -0.02, 0.28, -0.11, 0.07,
    0.13, -0.19, 0.13, -0.02, 149.62, 0
  ]
}
```

**Response:**
```json
{
  "prediction": 0,
  "fraud_probability": 0.00078,
  "fraud_score": 0,
  "is_fraud": false
}
```

#### POST `/predict`
Predict fraud using form data (for HTML forms)

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F 'features="[-1.36, -0.07, 2.54, ...]"'
```

### Analytics Endpoints

#### GET `/dashboard`
Get fraud detection statistics

**Response:**
```json
{
  "total_transactions": 150,
  "fraud_transactions": 12,
  "fraud_rate_percent": 8.0
}
```

#### GET `/history`
Get last 50 transactions

**Response:**
```json
[
  {
    "id": 1,
    "fraud_probability": 0.023,
    "fraud_score": 2,
    "is_fraud": false,
    "created_at": "2026-01-29 10:30:45"
  }
]
```

---

## 🧪 Testing

Run the test script:

```bash
python test_api.py
```

Or use the interactive Swagger UI:
```
http://127.0.0.1:8000/docs
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.95% |
| **Precision** | 95.2% |
| **Recall** | 87.3% |
| **F1-Score** | 91.1% |
| **ROC-AUC** | 98.7% |

**Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
**Training:** 284,807 transactions (492 fraudulent)  
**Technique:** SMOTE oversampling for class imbalance

---

## 📁 Project Structure

```
fraud-detection/
├── api/
│   ├── __init__.py
│   ├── app.py           # FastAPI application
│   ├── database.py      # SQLite setup
│   └── dashboard.py     # Analytics endpoints
├── notebook/
│   └── fraud_detection.ipynb  # Model training
├── data/
│   ├── creditcard.csv   # Dataset (not tracked)
│   └── fraud.db         # SQLite database
├── plots/               # Visualizations
├── fraud_model.pkl      # Trained XGBoost model
├── requirements.txt     # Dependencies
├── test_api.py          # API tests
└── README.md
```

---

## 🌐 Deployment

Deployed on [Render](https://render.com/) with automatic CI/CD from GitHub.

**Live API:** https://fraud-detection.onrender.com/docs

### Deploy Your Own

1. Fork this repository
2. Create a Render account
3. Connect your GitHub repo
4. Deploy with these settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn api.app:app --host 0.0.0.0 --port $PORT`
   - **Environment:** Python 3.12

---

## 🔮 Future Enhancements

- [ ] Add authentication (JWT tokens)
- [ ] PostgreSQL for production database
- [ ] Real-time monitoring dashboard
- [ ] Model versioning and A/B testing
- [ ] GraphQL API support
- [ ] Dockerization
- [ ] CI/CD pipeline with GitHub Actions

---

## 📝 License

This project is licensed under the MIT License.

---

## 👤 Author

**Your Name**  
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [Your Profile](https://linkedin.com/in/YOUR_PROFILE)

---

## 🙏 Acknowledgments

- Dataset: [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Machine Learning Research Lab, ULB

---

**⭐ If you find this project useful, please consider giving it a star!**
