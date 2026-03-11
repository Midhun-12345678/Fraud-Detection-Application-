import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import json
import os
from datetime import datetime

FEATURE_NAMES = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
                 "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
                 "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]


def run_drift_check(reference_csv_path: str, db_path: str) -> dict:
    """
    Run data drift detection comparing reference data to current (simulated) data.
    
    Args:
        reference_csv_path: Path to creditcard.csv
        db_path: Path to fraud.db SQLite database
    
    Returns:
        Dict with drift detection results
    """
    try:
        # Check if reference data exists
        if not os.path.exists(reference_csv_path):
            return {
                "error": "Training data not available",
                "drift_score": 0.0,
                "dataset_drift": False,
                "alert": False,
                "drifted_features": [],
                "n_drifted_features": 0,
                "total_features": 30,
                "timestamp": datetime.now().isoformat()
            }
        
        # Load reference data - sample 1000 rows
        reference_df = pd.read_csv(reference_csv_path)
        reference_sample = reference_df[FEATURE_NAMES].sample(n=min(1000, len(reference_df)), random_state=42)
        
        # Check transaction count in database
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM transactions")
        transaction_count = cursor.fetchone()[0]
        conn.close()
        
        # Simulate current data by sampling from CSV and adding noise
        current_sample = reference_df[FEATURE_NAMES].sample(n=min(1000, len(reference_df)), random_state=None)
        
        # Add Gaussian noise to simulate drift
        noise_multiplier = 2.0 if transaction_count < 50 else 1.0
        noise = np.random.normal(0, 0.5 * noise_multiplier, current_sample.shape)
        current_sample = current_sample + noise
        
        # Run Evidently drift report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_sample, current_data=current_sample)
        
        # Extract results from report
        report_dict = report.as_dict()
        
        # Navigate to drift results
        drift_results = None
        drifted_features = []
        
        for metric in report_dict.get("metrics", []):
            result = metric.get("result", {})
            
            # Check for dataset-level drift
            if "drift_share" in result:
                drift_results = result
            
            # Check for column-level drift
            if "drift_by_columns" in result:
                for col_name, col_data in result["drift_by_columns"].items():
                    if col_data.get("drift_detected", False):
                        drifted_features.append(col_name)
        
        if drift_results is None:
            # Fallback: try to find drift info in first metric
            drift_results = report_dict.get("metrics", [{}])[0].get("result", {})
        
        # Extract key metrics
        drift_score = drift_results.get("drift_share", 0.0)
        dataset_drift = drift_results.get("dataset_drift", False)
        n_drifted = drift_results.get("number_of_drifted_columns", len(drifted_features))
        
        # Build result
        result = {
            "timestamp": datetime.now().isoformat(),
            "dataset_drift": bool(dataset_drift),
            "drift_score": float(drift_score),
            "drifted_features": drifted_features,
            "n_drifted_features": int(n_drifted),
            "total_features": 30,
            "alert": drift_score > 0.3
        }
        
        # Save to history
        save_drift_report(result)
        
        return result
        
    except Exception as e:
        return {
            "error": str(e),
            "drift_score": 0.0,
            "dataset_drift": False,
            "alert": False,
            "drifted_features": [],
            "n_drifted_features": 0,
            "total_features": 30,
            "timestamp": datetime.now().isoformat()
        }


def save_drift_report(result: dict, path: str = "data/drift_log.json") -> None:
    """
    Append drift result to JSON log file.
    
    Args:
        result: Drift check result dict
        path: Path to drift log file
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Load existing history or create empty list
        if os.path.exists(path):
            with open(path, "r") as f:
                history = json.load(f)
        else:
            history = []
        
        # Append new result
        history.append(result)
        
        # Save back
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
            
    except Exception as e:
        print(f"Error saving drift report: {e}")


def get_drift_history(path: str = "data/drift_log.json") -> list:
    """
    Get last 20 drift check results.
    
    Args:
        path: Path to drift log file
    
    Returns:
        List of last 20 drift results
    """
    try:
        if not os.path.exists(path):
            return []
        
        with open(path, "r") as f:
            history = json.load(f)
        
        # Return last 20 results
        return history[-20:]
        
    except Exception as e:
        print(f"Error loading drift history: {e}")
        return []
