#!/usr/bin/env python3
import os
import sys

def main():
    print("=" * 60)
    print("🚀 MLOps Auto-Retrain System")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists('models/model.pkl'):
        print("\n⚠️  No model found. Training initial model...")
        print("\nRun: python ml_model/train.py")
        print("Then restart this server.\n")
        sys.exit(1)
    
    print("\n✅ Model found")
    print("\n📡 Starting API server...")
    print("\nEndpoints:")
    print("  • POST /predict       - Make predictions")
    print("  • GET  /check_drift   - Check for drift")
    print("  • POST /retrain       - Trigger retraining")
    print("  • GET  /stats         - System stats")
    print("\n" + "=" * 60)
    print("Server running at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    print("=" * 60 + "\n")
    
    os.system("uvicorn mlops_system.api:app --host 0.0.0.0 --port 8000 --reload")

if __name__ == "__main__":
    main()
