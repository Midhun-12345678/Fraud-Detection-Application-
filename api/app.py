from fastapi import FastAPI, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import json
import os
import shap

from .database import conn, cursor
from .dashboard import router

# Feature names for SHAP explanations
FEATURE_NAMES = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

app = FastAPI()

# Mount static files
static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Register dashboard routes
app.include_router(router)

# Load model with proper path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_root, "fraud_model.pkl")
model = joblib.load(model_path)

def reload_model():
    """Reload the model from disk after retraining."""
    global model, explainer
    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)

# Initialize SHAP explainer for model interpretability
explainer = shap.TreeExplainer(model)

class TransactionFeatures(BaseModel):
    features: list[float]

# Serve index.html at root
@app.get("/", response_class=FileResponse)
def read_root():
    return os.path.join(static_path, "index.html")

@app.post("/predict")
async def predict(features: str = Form(...)):
    try:
        feature_list = json.loads(features)
        feature_array = np.array(feature_list).reshape(1, -1)

        prediction = model.predict(feature_array)[0]
        probability = model.predict_proba(feature_array)[0][1]

        fraud_score = int(probability * 100)
        is_fraud = int(prediction == 1)

        cursor.execute(
            "INSERT INTO transactions (fraud_probability, fraud_score, is_fraud) VALUES (?, ?, ?)",
            (float(probability), fraud_score, is_fraud)
        )
        conn.commit()

        return {
            "prediction": int(prediction),
            "fraud_probability": float(probability),
            "fraud_score": fraud_score,
            "is_fraud": bool(is_fraud)
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in features")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-json")
async def predict_json(data: TransactionFeatures):
    try:
        feature_array = np.array(data.features).reshape(1, -1)

        prediction = model.predict(feature_array)[0]
        probability = model.predict_proba(feature_array)[0][1]

        fraud_score = int(probability * 100)
        is_fraud = int(prediction == 1)

        # Compute SHAP values for explainability
        shap_values = explainer.shap_values(feature_array)
        # For binary classification, get values for positive class (fraud)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]  # Class 1 (fraud) SHAP values
        else:
            shap_vals = shap_values[0]  # Single output format
        
        # Get top 5 features by absolute SHAP value
        feature_importance = list(zip(FEATURE_NAMES, shap_vals))
        top_features = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)[:5]
        top_features_list = [{"feature": f, "shap_value": float(v)} for f, v in top_features]

        cursor.execute(
            "INSERT INTO transactions (fraud_probability, fraud_score, is_fraud) VALUES (?, ?, ?)",
            (float(probability), fraud_score, is_fraud)
        )
        conn.commit()

        return {
            "prediction": int(prediction),
            "fraud_probability": float(probability),
            "fraud_score": fraud_score,
            "is_fraud": bool(is_fraud),
            "shap_values": [float(v) for v in shap_vals],
            "top_features": top_features_list
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))