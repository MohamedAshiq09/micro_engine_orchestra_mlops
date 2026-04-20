import numpy as np
from scipy import stats
from typing import Dict, Tuple

class DriftDetector:
    def __init__(self, threshold: float = 0.05):
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
        Detect drift using Kolmogorov-Smirnov test
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
            # Generate baseline distribution
            baseline_samples = np.random.normal(
                self.baseline_stats['mean'][i],
                self.baseline_stats['std'][i],
                size=len(new_data)
            )
            
            # KS test
            statistic, p_value = stats.ks_2samp(baseline_samples, new_data[:, i])
            
            feature_drift = p_value < self.threshold
            drift_info['features'][f'feature{i+1}'] = {
                'p_value': float(p_value),
                'drift': feature_drift
            }
            
            if feature_drift:
                drift_detected = True
        
        drift_info['overall_drift'] = drift_detected
        return drift_detected, drift_info
