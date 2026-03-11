import pandas as pd
import numpy as np
import joblib
import json
import os
import shutil
import time
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_NAMES = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
                 "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
                 "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate model on test data."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba))
    }


def train_new_model(csv_path: str) -> tuple:
    """Train a new XGBoost model."""
    logger.info(f"Loading training data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    X = df[FEATURE_NAMES]
    y = df["Class"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # SMOTE on training data only
    logger.info("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    logger.info(f"After SMOTE: {len(X_train_resampled)} samples")
    
    # Train XGBoost
    logger.info("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )
    model.fit(X_train_resampled, y_train_resampled)
    
    metrics = evaluate_model(model, X_test, y_test)
    logger.info(f"New model metrics: {metrics}")
    
    return model, metrics, X_test, y_test


def should_retrain(drift_score: float, current_accuracy: float = None, new_accuracy_threshold: float = 0.94) -> tuple:
    """Determine if retraining is needed."""
    if drift_score > 0.3:
        return (True, f"Data drift detected: {drift_score:.2%} > 30% threshold")
    
    if current_accuracy is not None and current_accuracy < new_accuracy_threshold:
        return (True, f"Model accuracy degraded: {current_accuracy:.2%} < {new_accuracy_threshold:.0%} threshold")
    
    return (False, "Model performing well, no retraining needed")


def run_retraining_pipeline(csv_path: str, model_path: str, backup: bool = True) -> dict:
    """Main retraining pipeline."""
    try:
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        logger.info("=" * 50)
        logger.info("RETRAINING PIPELINE STARTED")
        logger.info("=" * 50)
        
        # Load current model
        current_model = joblib.load(model_path)
        
        # Load sample for evaluation
        df = pd.read_csv(csv_path)
        sample = df.sample(n=min(10000, len(df)), random_state=42)
        X_sample = sample[FEATURE_NAMES]
        y_sample = sample["Class"]
        
        # Evaluate current model
        current_metrics = evaluate_model(current_model, X_sample, y_sample)
        logger.info(f"Current model metrics: {current_metrics}")
        
        # Check drift
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from drift.drift_detector import run_drift_check
        
        drift_result = run_drift_check(csv_path, "data/fraud.db")
        drift_score = drift_result.get("drift_score", 0.0)
        logger.info(f"Drift score: {drift_score:.2%}")
        
        # Check if retraining needed
        retrain_needed, reason = should_retrain(drift_score, current_metrics["accuracy"])
        
        if not retrain_needed:
            logger.info(f"Retraining skipped: {reason}")
            result = {
                "status": "skipped",
                "reason": reason,
                "timestamp": timestamp,
                "current_metrics": current_metrics,
                "new_metrics": None,
                "drift_score": drift_score,
                "promoted": False,
                "training_duration_seconds": time.time() - start_time
            }
            save_retrain_log(result)
            return result
        
        logger.info(f"Retraining triggered: {reason}")
        
        # Backup current model
        if backup:
            backup_name = f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            backup_path = os.path.join("data", backup_name)
            logger.info(f"Backing up to {backup_path}")
            shutil.copy2(model_path, backup_path)
        
        # Train new model
        new_model, new_metrics, X_test, y_test = train_new_model(csv_path)
        
        # Compare models on F1
        current_f1 = current_metrics["f1"]
        new_f1 = new_metrics["f1"]
        
        if new_f1 > current_f1:
            logger.info(f"New model promoted: F1 {current_f1:.4f} -> {new_f1:.4f}")
            joblib.dump(new_model, model_path)
            status = "promoted"
            promoted = True
        else:
            logger.info(f"New model rejected: F1 {new_f1:.4f} < {current_f1:.4f}")
            status = "rejected"
            promoted = False
        
        training_duration = time.time() - start_time
        
        result = {
            "status": status,
            "reason": reason,
            "timestamp": timestamp,
            "current_metrics": current_metrics,
            "new_metrics": new_metrics,
            "drift_score": drift_score,
            "promoted": promoted,
            "training_duration_seconds": training_duration
        }
        
        save_retrain_log(result)
        logger.info(f"RETRAINING COMPLETED: {status.upper()} in {training_duration:.1f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        result = {
            "status": "error",
            "reason": str(e),
            "timestamp": datetime.now().isoformat(),
            "current_metrics": None,
            "new_metrics": None,
            "drift_score": 0.0,
            "promoted": False,
            "training_duration_seconds": 0.0
        }
        save_retrain_log(result)
        return result


def save_retrain_log(result: dict, path: str = "data/retrain_log.json") -> None:
    """Append result to retrain log."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if os.path.exists(path):
            with open(path, "r") as f:
                history = json.load(f)
        else:
            history = []
        
        history.append(result)
        
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving retrain log: {e}")


def get_retrain_history(path: str = "data/retrain_log.json") -> list:
    """Get last 10 retraining results."""
    try:
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            history = json.load(f)
        return history[-10:]
    except Exception as e:
        logger.error(f"Error loading retrain history: {e}")
        return []


def get_current_metrics(csv_path: str, model_path: str, sample_size: int = 5000) -> dict:
    """Get current model metrics."""
    try:
        model = joblib.load(model_path)
        df = pd.read_csv(csv_path)
        sample = df.sample(n=min(sample_size, len(df)), random_state=42)
        X_sample = sample[FEATURE_NAMES]
        y_sample = sample["Class"]
        return evaluate_model(model, X_sample, y_sample)
    except Exception as e:
        logger.error(f"Error getting current metrics: {e}")
        return {"error": str(e)}
