import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

class ModelManager:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.train_stats = None
        os.makedirs(model_dir, exist_ok=True)
        
    def load_model(self):
        """Load the current model"""
        model_path = os.path.join(self.model_dir, 'model.pkl')
        stats_path = os.path.join(self.model_dir, 'train_stats.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model not found. Train initial model first.")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                self.train_stats = pickle.load(f)
        
        return self.model
    
    def predict(self, features: np.ndarray) -> float:
        """Make prediction"""
        if self.model is None:
            self.load_model()
        
        prediction = self.model.predict(features)
        return float(prediction[0])
    
    def retrain(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Retrain model with new data"""
        print(f"🔄 Retraining model with {len(X)} samples...")
        
        # Train new model
        new_model = RandomForestRegressor(n_estimators=100, random_state=42)
        new_model.fit(X, y)
        
        # Calculate score
        score = new_model.score(X, y)
        
        # Backup old model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        old_model_path = os.path.join(self.model_dir, f'model_backup_{timestamp}.pkl')
        
        if os.path.exists(os.path.join(self.model_dir, 'model.pkl')):
            os.rename(
                os.path.join(self.model_dir, 'model.pkl'),
                old_model_path
            )
        
        # Save new model
        with open(os.path.join(self.model_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(new_model, f)
        
        # Update statistics
        train_stats = {
            'mean': {f'feature{i+1}': float(X[:, i].mean()) for i in range(X.shape[1])},
            'std': {f'feature{i+1}': float(X[:, i].std()) for i in range(X.shape[1])}
        }
        
        with open(os.path.join(self.model_dir, 'train_stats.pkl'), 'wb') as f:
            pickle.dump(train_stats, f)
        
        # Reload model
        self.model = new_model
        self.train_stats = train_stats
        
        print(f"✅ Model retrained successfully! Score: {score:.4f}")
        
        return {
            'status': 'success',
            'score': float(score),
            'samples': len(X),
            'timestamp': timestamp
        }
