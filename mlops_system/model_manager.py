import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from datetime import datetime

class ModelManager:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.train_stats = None
        self.previous_model = None
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
    
    def predict(self, features: np.ndarray) -> tuple:
        """Make prediction with confidence"""
        if self.model is None:
            self.load_model()
        
        prediction = self.model.predict(features)
        
        # Calculate confidence (simple approach)
        if self.train_stats:
            baseline_mean = np.mean(list(self.train_stats['mean'].values()))
            confidence = 1.0 / (1.0 + abs(prediction[0] - baseline_mean) / baseline_mean)
            confidence = min(max(confidence, 0.5), 0.99)  # Clamp between 0.5 and 0.99
        else:
            confidence = 0.85
        
        return float(prediction[0]), float(confidence)
    
    def compare_models(self, old_model, new_model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Compare old vs new model performance"""
        old_pred = old_model.predict(X_test)
        new_pred = new_model.predict(X_test)
        
        old_score = r2_score(y_test, old_pred)
        new_score = r2_score(y_test, new_pred)
        
        old_mae = mean_absolute_error(y_test, old_pred)
        new_mae = mean_absolute_error(y_test, new_pred)
        
        improvement = ((new_score - old_score) / abs(old_score)) * 100 if old_score != 0 else 0
        
        return {
            'old_score': float(old_score),
            'new_score': float(new_score),
            'old_mae': float(old_mae),
            'new_mae': float(new_mae),
            'improvement': float(improvement),
            'deploy': new_score > old_score
        }
    
    def retrain(self, X: np.ndarray, y: np.ndarray, min_improvement: float = 0.0) -> dict:
        """Retrain model with validation"""
        print(f"🔄 Retraining model with {len(X)} samples...")
        
        try:
            # Backup current model
            self.previous_model = self.model
            
            # Split for validation
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train new model
            new_model = RandomForestRegressor(n_estimators=100, random_state=42)
            new_model.fit(X_train, y_train)
            
            # Calculate scores
            train_score = new_model.score(X_train, y_train)
            val_score = new_model.score(X_val, y_val)
            
            # Compare with old model if exists
            comparison = None
            deploy = True
            
            if self.previous_model is not None:
                comparison = self.compare_models(self.previous_model, new_model, X_val, y_val)
                deploy = comparison['deploy'] and comparison['improvement'] >= min_improvement
                
                if not deploy:
                    print(f"⚠️  New model not better. Keeping current model.")
                    return {
                        'status': 'rejected',
                        'reason': 'No improvement',
                        'comparison': comparison
                    }
            
            # Save new model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            versioned_path = os.path.join(self.model_dir, f'model_{timestamp}.pkl')
            
            with open(versioned_path, 'wb') as f:
                pickle.dump(new_model, f)
            
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
            
            print(f"✅ Model retrained! Train: {train_score:.4f}, Val: {val_score:.4f}")
            
            return {
                'status': 'success',
                'train_score': float(train_score),
                'val_score': float(val_score),
                'samples': len(X),
                'timestamp': timestamp,
                'comparison': comparison
            }
            
        except Exception as e:
            # Rollback on failure
            print(f"❌ Retraining failed: {e}")
            if self.previous_model is not None:
                self.model = self.previous_model
                print("✅ Rolled back to previous model")
            
            return {
                'status': 'failed',
                'error': str(e)
            }
