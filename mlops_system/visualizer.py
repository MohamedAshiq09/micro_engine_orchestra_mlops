import matplotlib.pyplot as plt
import numpy as np
import os

class DriftVisualizer:
    def __init__(self, output_dir='logs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def plot_drift(self, baseline_data: np.ndarray, new_data: np.ndarray, feature_idx: int = 0):
        """Plot distribution comparison"""
        fig, ax = plt.subplots(figsize=(10, 4))
        
        ax.hist(baseline_data[:, feature_idx], bins=30, alpha=0.6, 
                label='Baseline', color='#3498db', edgecolor='black')
        ax.hist(new_data[:, feature_idx], bins=30, alpha=0.6, 
                label='Current', color='#e74c3c', edgecolor='black')
        
        ax.set_xlabel(f'Feature {feature_idx + 1}', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Data Distribution Shift', fontsize=14, fontweight='light')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_path = os.path.join(self.output_dir, f'drift_feature_{feature_idx}.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return output_path
