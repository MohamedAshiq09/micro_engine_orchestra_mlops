import json
import os
from datetime import datetime
from typing import List, Dict
import numpy as np

class DataStore:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 'predictions.jsonl')
        
    def log_prediction(self, features: Dict, prediction: float):
        """Log incoming prediction data"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'prediction': prediction
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_recent_data(self, n: int = 100) -> List[Dict]:
        """Get recent logged data"""
        if not os.path.exists(self.log_file):
            return []
        
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
        
        recent_lines = lines[-n:] if len(lines) > n else lines
        return [json.loads(line) for line in recent_lines]
    
    def get_feature_matrix(self, data: List[Dict]) -> np.ndarray:
        """Convert logged data to feature matrix"""
        if not data:
            return np.array([])
        
        features = [list(entry['features'].values()) for entry in data]
        return np.array(features)
