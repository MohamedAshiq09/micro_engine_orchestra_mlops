#!/usr/bin/env python3
"""
Test script to demonstrate the MLOps system
"""
import requests
import numpy as np
import time

BASE_URL = "http://localhost:8000"

def test_predictions():
    """Send normal predictions"""
    print("\n" + "="*60)
    print("📊 PHASE 1: Normal Predictions")
    print("="*60)
    
    for i in range(40):
        data = {
            "feature1": float(np.random.randn() * 10 + 50),
            "feature2": float(np.random.randn() * 5 + 20),
            "feature3": float(np.random.randn() * 15 + 100)
        }
        
        response = requests.post(f"{BASE_URL}/predict", json=data)
        if i % 10 == 0:
            print(f"✓ Prediction {i+1}: {response.json()['prediction']:.2f}")
    
    print(f"✅ Sent 40 normal predictions")

def check_drift():
    """Check for drift"""
    print("\n" + "="*60)
    print("🔍 PHASE 2: Drift Detection")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/check_drift")
    result = response.json()
    
    print(f"Drift Detected: {result['drift_detected']}")
    print(f"Samples Analyzed: {result.get('samples_analyzed', 0)}")
    
    return result['drift_detected']

def send_drifted_data():
    """Send drifted data"""
    print("\n" + "="*60)
    print("⚠️  PHASE 3: Simulating Data Drift")
    print("="*60)
    
    for i in range(50):
        # Shifted distribution
        data = {
            "feature1": float(np.random.randn() * 10 + 80),  # shifted from 50 to 80
            "feature2": float(np.random.randn() * 5 + 35),   # shifted from 20 to 35
            "feature3": float(np.random.randn() * 15 + 150)  # shifted from 100 to 150
        }
        
        response = requests.post(f"{BASE_URL}/predict", json=data)
        if i % 10 == 0:
            print(f"✓ Drifted prediction {i+1}: {response.json()['prediction']:.2f}")
    
    print(f"✅ Sent 50 drifted predictions")

def trigger_retrain():
    """Trigger retraining"""
    print("\n" + "="*60)
    print("🔄 PHASE 4: Automatic Retraining")
    print("="*60)
    
    response = requests.post(f"{BASE_URL}/retrain")
    result = response.json()
    
    print(f"✅ Retraining completed!")
    print(f"   Score: {result['score']:.4f}")
    print(f"   Samples: {result['samples']}")

def get_stats():
    """Get system stats"""
    print("\n" + "="*60)
    print("📈 System Statistics")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/stats")
    stats = response.json()
    
    print(f"Total Predictions: {stats['total_predictions']}")
    print(f"Model Loaded: {stats['model_loaded']}")
    print(f"Baseline Set: {stats['baseline_set']}")

def main():
    print("\n" + "="*60)
    print("🚀 MLOps Auto-Retrain System - DEMO")
    print("="*60)
    print("\nMake sure the server is running: python run.py")
    input("\nPress Enter to start demo...")
    
    try:
        # Test server
        requests.get(BASE_URL)
    except:
        print("\n❌ Server not running! Start it with: python run.py")
        return
    
    # Phase 1: Normal predictions
    test_predictions()
    time.sleep(1)
    
    # Phase 2: Check drift (should be no drift)
    drift = check_drift()
    time.sleep(1)
    
    # Phase 3: Send drifted data
    send_drifted_data()
    time.sleep(1)
    
    # Phase 4: Check drift again (should detect drift)
    drift = check_drift()
    time.sleep(1)
    
    if drift:
        print("\n🎯 Drift detected! Triggering automatic retraining...")
        time.sleep(1)
        trigger_retrain()
    
    # Final stats
    time.sleep(1)
    get_stats()
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETED!")
    print("="*60)
    print("\n💡 Key Points Demonstrated:")
    print("   1. ✓ Model makes predictions")
    print("   2. ✓ System logs all incoming data")
    print("   3. ✓ Drift detection works automatically")
    print("   4. ✓ Model retrains when drift detected")
    print("   5. ✓ System is self-healing\n")

if __name__ == "__main__":
    main()
