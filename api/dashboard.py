from fastapi import APIRouter
from .database import cursor

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