from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import numpy as np
from typing import Dict
import sys
import os
import pickle
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from mlops_system.model_manager import ModelManager
from mlops_system.drift_detector import DriftDetector
from mlops_system.data_store import DataStore
from mlops_system.version_manager import VersionManager

app = FastAPI(title="MLOps Auto-Retrain System")

# Initialize components
model_manager = ModelManager()
drift_detector = DriftDetector(threshold=0.3)
data_store = DataStore()
version_manager = VersionManager()

# Configuration
DRIFT_THRESHOLD = 0.3
MIN_IMPROVEMENT = 0.0  # Minimum improvement % to deploy new model
AUTO_RETRAIN = True

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
    confidence: float
    drift_detected: bool

class TrainModelRequest(BaseModel):
    model_type: str
    n_samples: int = 1000

@app.get("/")
def root():
    return {
        "message": "MLOps Auto-Retrain System",
        "version": version_manager.get_latest_version(),
        "endpoints": {
            "/predict": "Make predictions",
            "/check_drift": "Check for data drift",
            "/retrain": "Trigger retraining",
            "/stats": "System statistics",
            "/versions": "Model versions",
            "/config": "System configuration"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Make prediction with confidence"""
    try:
        features_dict = {
            'feature1': request.feature1,
            'feature2': request.feature2,
            'feature3': request.feature3
        }
        
        features_array = np.array([[request.feature1, request.feature2, request.feature3]])
        prediction, confidence = model_manager.predict(features_array)
        
        # Log prediction
        data_store.log_prediction(features_dict, prediction)
        
        # Check if we should auto-retrain
        drift_detected = False
        if AUTO_RETRAIN:
            recent_data = data_store.get_recent_data(n=100)
            if len(recent_data) >= 50:
                feature_matrix = data_store.get_feature_matrix(recent_data)
                drift_detected, _ = drift_detector.detect_drift(feature_matrix)
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "drift_detected": drift_detected
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
            "drift_info": drift_info,
            "threshold": DRIFT_THRESHOLD
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
def retrain():
    """Retrain model with validation"""
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
        
        # Retrain with validation
        result = model_manager.retrain(X, y, min_improvement=MIN_IMPROVEMENT)
        
        if result['status'] == 'success':
            # Update baseline
            drift_detector.set_baseline(X)
            
            # Save version
            version = version_manager.save_version(
                model_manager.model,
                result['val_score'],
                result['samples'],
                result.get('comparison')
            )
            result['version'] = version
        
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
        latest_version = version_manager.get_latest_version()
        
        return {
            "total_predictions": len(recent_data),
            "model_loaded": model_manager.model is not None,
            "baseline_set": drift_detector.baseline_stats is not None,
            "total_versions": version_manager.get_version_count(),
            "latest_version": latest_version,
            "auto_retrain": AUTO_RETRAIN,
            "drift_threshold": DRIFT_THRESHOLD
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/versions")
def get_versions():
    """Get all model versions"""
    try:
        versions = version_manager.load_versions()
        return {
            "total": len(versions),
            "versions": versions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
def get_config():
    """Get system configuration"""
    return {
        "drift_threshold": DRIFT_THRESHOLD,
        "min_improvement": MIN_IMPROVEMENT,
        "auto_retrain": AUTO_RETRAIN
    }

@app.post("/rollback/{version}")
def rollback_version(version: str):
    """Rollback to specific version"""
    try:
        success = version_manager.rollback(version)
        if success:
            model_manager.load_model()
            return {"status": "success", "version": version}
        else:
            raise HTTPException(status_code=404, detail="Version not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...)):
    """Upload and deploy a trained model"""
    try:
        if not file.filename.endswith('.pkl'):
            raise HTTPException(status_code=400, detail="Only .pkl files allowed")
        
        # Read uploaded file
        contents = await file.read()
        
        # Validate it's a pickle file
        try:
            model = pickle.loads(contents)
        except:
            raise HTTPException(status_code=400, detail="Invalid pickle file")
        
        # Check if model has predict method
        if not hasattr(model, 'predict'):
            raise HTTPException(status_code=400, detail="Model must have predict() method")
        
        # Save model
        model_path = os.path.join('models', 'model.pkl')
        with open(model_path, 'wb') as f:
            f.write(contents)
        
        # Reload model manager
        model_manager.load_model()
        
        # Create version
        version = version_manager.save_version(
            model_manager.model,
            0.0,  # Score unknown for uploaded models
            0,
            {'source': 'uploaded'}
        )
        
        print(f"✅ Model uploaded: {file.filename}")
        
        return {
            "status": "success",
            "filename": file.filename,
            "version": version,
            "message": "Model deployed successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_new_model")
def train_new_model(request: TrainModelRequest):
    """Train a new model from scratch"""
    try:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        
        # Generate synthetic data
        np.random.seed(42)
        n = request.n_samples
        
        X = np.column_stack([
            np.random.randn(n) * 10 + 50,
            np.random.randn(n) * 5 + 20,
            np.random.randn(n) * 15 + 100
        ])
        
        y = X[:, 0] * 0.5 + X[:, 1] * 1.2 + X[:, 2] * 0.3 + np.random.randn(n) * 5
        
        # Select model type
        if request.model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif request.model_type == "Linear Regression":
            model = LinearRegression()
        elif request.model_type == "Gradient Boosting":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")
        
        # Train
        model.fit(X, y)
        score = model.score(X, y)
        
        # Save model
        model_path = os.path.join('models', 'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save stats
        train_stats = {
            'mean': {f'feature{i+1}': float(X[:, i].mean()) for i in range(3)},
            'std': {f'feature{i+1}': float(X[:, i].std()) for i in range(3)}
        }
        
        with open(os.path.join('models', 'train_stats.pkl'), 'wb') as f:
            pickle.dump(train_stats, f)
        
        # Reload
        model_manager.load_model()
        
        # Set baseline
        drift_detector.set_baseline(X)
        
        # Create version
        version = version_manager.save_version(
            model,
            score,
            n,
            {'model_type': request.model_type}
        )
        
        print(f"✅ New model trained: {request.model_type}")
        
        return {
            "status": "success",
            "model_type": request.model_type,
            "score": float(score),
            "samples": n,
            "version": version
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
