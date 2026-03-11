from fastapi import APIRouter
from .database import cursor
import sys
import os

# Add project root to path for drift imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift.drift_detector import run_drift_check, get_drift_history
from retrain.retrain_pipeline import run_retraining_pipeline, get_retrain_history, get_current_metrics

router = APIRouter()

@router.get("/dashboard")
def dashboard():
    cursor.execute("SELECT COUNT(*) FROM transactions")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud = 1")
    frauds = cursor.fetchone()[0]

    return {
        "total_transactions": total,
        "fraud_transactions": frauds,
        "fraud_rate_percent": round((frauds / max(total, 1)) * 100, 2)
    }

@router.get("/history")
def history():
    cursor.execute(
        "SELECT id, fraud_probability, fraud_score, is_fraud, created_at FROM transactions ORDER BY id DESC LIMIT 50"
    )
    rows = cursor.fetchall()

    return [
        {
            "id": r[0],
            "fraud_probability": r[1],
            "fraud_score": r[2],
            "is_fraud": bool(r[3]),
            "created_at": r[4]
        }
        for r in rows
    ]

@router.get("/drift")
def check_drift():
    """Run drift detection on current vs reference data."""
    result = run_drift_check("data/creditcard.csv", "data/fraud.db")
    return result

@router.get("/drift/history")
def drift_history():
    """Get last 20 drift check results."""
    return get_drift_history()

@router.post("/retrain")
def trigger_retrain():
    """Trigger model retraining pipeline."""
    result = run_retraining_pipeline("data/creditcard.csv", "fraud_model.pkl")
    
    # Reload model if promoted
    if result.get("status") == "promoted":
        from .app import reload_model
        reload_model()
    
    return result

@router.get("/retrain/status")
def retrain_status():
    """Get retraining history."""
    return get_retrain_history()

@router.get("/retrain/current-metrics")
def current_metrics():
    """Get current model metrics."""
    return get_current_metrics("data/creditcard.csv", "fraud_model.pkl", sample_size=5000)