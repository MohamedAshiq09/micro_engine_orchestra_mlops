import numpy as np
from typing import Dict, Tuple

class DriftDetector:
    def __init__(self, threshold: float = 0.3):
        """
        threshold: percentage change threshold (0.3 = 30% change triggers drift)
        """
        self.threshold = threshold
        self.baseline_stats = None
        
    def set_baseline(self, data: np.ndarray):
        """Set baseline statistics from training data"""
        self.baseline_stats = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0)
        }
    
    def detect_drift(self, new_data: np.ndarray) -> Tuple[bool, Dict]:
        """
        Detect drift using statistical distance
        Returns: (drift_detected, drift_info)
        """
        if self.baseline_stats is None:
            return False, {'error': 'Baseline not set'}
        
        if len(new_data) < 30:
            return False, {'message': 'Insufficient data for drift detection'}
        
        drift_detected = False
        drift_info = {'features': {}}
        
        # Check each feature
        for i in range(new_data.shape[1]):
            new_mean = np.mean(new_data[:, i])
            new_std = np.std(new_data[:, i])
            
            baseline_mean = self.baseline_stats['mean'][i]
            baseline_std = self.baseline_stats['std'][i]
            
            # Calculate percentage change in mean
            mean_change = abs(new_mean - baseline_mean) / (abs(baseline_mean) + 1e-10)
            
            # Calculate percentage change in std
            std_change = abs(new_std - baseline_std) / (abs(baseline_std) + 1e-10)
            
            # Drift if either mean or std changed significantly
            feature_drift = (mean_change > self.threshold) or (std_change > self.threshold)
            
            drift_info['features'][f'feature{i+1}'] = {
                'mean_change': float(mean_change),
                'std_change': float(std_change),
                'drift': bool(feature_drift)
            }
            
            if feature_drift:
                drift_detected = True
        
        drift_info['overall_drift'] = bool(drift_detected)
        return bool(drift_detected), drift_info
