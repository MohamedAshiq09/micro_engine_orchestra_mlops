import json
import os
import pickle
import shutil
from datetime import datetime
from typing import Dict, List, Optional

class VersionManager:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.versions_file = os.path.join(model_dir, 'versions.json')
        os.makedirs(model_dir, exist_ok=True)
        
    def save_version(self, model, score: float, samples: int, metrics: Dict = None) -> str:
        """Save model with version"""
        version = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(self.model_dir, f'model_{version}.pkl')
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save version metadata
        version_info = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'score': float(score),
            'samples': samples,
            'metrics': metrics or {},
            'model_path': model_path
        }
        
        # Update versions file
        versions = self.load_versions()
        versions.append(version_info)
        
        with open(self.versions_file, 'w') as f:
            json.dump(versions, f, indent=2)
        
        # Update current model symlink
        current_path = os.path.join(self.model_dir, 'model.pkl')
        if os.path.exists(current_path):
            os.remove(current_path)
        shutil.copy(model_path, current_path)
        
        return version
    
    def load_versions(self) -> List[Dict]:
        """Load all version metadata"""
        if not os.path.exists(self.versions_file):
            return []
        
        with open(self.versions_file, 'r') as f:
            return json.load(f)
    
    def get_latest_version(self) -> Optional[Dict]:
        """Get latest version info"""
        versions = self.load_versions()
        return versions[-1] if versions else None
    
    def rollback(self, version: str) -> bool:
        """Rollback to specific version"""
        versions = self.load_versions()
        
        for v in versions:
            if v['version'] == version:
                model_path = v['model_path']
                current_path = os.path.join(self.model_dir, 'model.pkl')
                
                if os.path.exists(model_path):
                    shutil.copy(model_path, current_path)
                    print(f"✅ Rolled back to version {version}")
                    return True
        
        return False
    
    def get_version_count(self) -> int:
        """Get total number of versions"""
        return len(self.load_versions())
