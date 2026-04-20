from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from mlops_system.model_manager import ModelManager
from mlops_system.drift_detector import DriftDetector
from mlops_system.data_store import DataStore

app = FastAPI(title="MLOps Auto-Retrain System")

# Initialize components
model_manager = ModelManager()
drift_detector = DriftDetector(threshold=0.3)  # 30% change triggers drift
data_store = DataStore()

# Load model and set baseline
try:
    model_manager.load_model()
    if model_manager.train_stats:
        baseline_data = np.array([
            [model_manager.train_stats['mean'][f'feature{i+1}'] 
             for i in range(3)]
        ])
        drift_detector.set_baseline(baseline_data)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️  Model not loaded: {e}")

class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float

class PredictionResponse(BaseModel):
    prediction: float
    drift_detected: bool
    drift_info: Dict

@app.get("/")
def root():
    return {
        "message": "MLOps Auto-Retrain System",
        "endpoints": {
            "/predict": "Make predictions",
            "/check_drift": "Check for data drift",
            "/retrain": "Trigger retraining",
            "/stats": "System statistics"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Make prediction and log data"""
    try:
        features_dict = {
            'feature1': request.feature1,
            'feature2': request.feature2,
            'feature3': request.feature3
        }
        
        features_array = np.array([[request.feature1, request.feature2, request.feature3]])
        prediction = model_manager.predict(features_array)
        
        # Log prediction
        data_store.log_prediction(features_dict, prediction)
        
        return {
            "prediction": prediction,
            "drift_detected": False,
            "drift_info": {"message": "Use /check_drift endpoint"}
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/check_drift")
def check_drift():
    """Check for drift in recent data"""
    try:
        recent_data = data_store.get_recent_data(n=100)
        
        if len(recent_data) < 30:
            return {
                "drift_detected": False,
                "message": f"Insufficient data ({len(recent_data)}/30 required)"
            }
        
        feature_matrix = data_store.get_feature_matrix(recent_data)
        drift_detected, drift_info = drift_detector.detect_drift(feature_matrix)
        
        if drift_detected:
            print("⚠️  DRIFT DETECTED!")
        
        return {
            "drift_detected": drift_detected,
            "samples_analyzed": len(recent_data),
            "drift_info": drift_info
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
def retrain():
    """Retrain model with recent data"""
    try:
        recent_data = data_store.get_recent_data(n=200)
        
        if len(recent_data) < 50:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for retraining ({len(recent_data)}/50 required)"
            )
        
        # Prepare training data
        X = data_store.get_feature_matrix(recent_data)
        y = np.array([entry['prediction'] for entry in recent_data])
        
        # Retrain
        result = model_manager.retrain(X, y)
        
        # Update baseline
        drift_detector.set_baseline(X)
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_stats():
    """Get system statistics"""
    try:
        recent_data = data_store.get_recent_data(n=1000)
        
        return {
            "total_predictions": len(recent_data),
            "model_loaded": model_manager.model is not None,
            "baseline_set": drift_detector.baseline_stats is not None
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
