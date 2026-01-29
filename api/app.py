from fastapi import FastAPI, Form, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import json
import os

from .database import conn, cursor
from .dashboard import router

app = FastAPI()

# Register dashboard routes
app.include_router(router)

# Load model with proper path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_root, "fraud_model.pkl")
model = joblib.load(model_path)

class TransactionFeatures(BaseModel):
    features: list[float]

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is running"}