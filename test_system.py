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
        result = response.json()
        
        if i % 10 == 0:
            print(f"✓ Prediction {i+1}: {result['prediction']:.2f} (Confidence: {result['confidence']*100:.1f}%)")
    
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
    print("🔄 PHASE 4: Automatic Retraining with Validation")
    print("="*60)
    
    response = requests.post(f"{BASE_URL}/retrain")
    result = response.json()
    
    if result['status'] == 'success':
        print(f"✅ Retraining completed!")
        print(f"   Train Score: {result['train_score']:.4f}")
        print(f"   Val Score: {result['val_score']:.4f}")
        print(f"   Samples: {result['samples']}")
        
        if result.get('comparison'):
            comp = result['comparison']
            print(f"\n📊 Performance Comparison:")
            print(f"   Old Model: {comp['old_score']:.4f}")
            print(f"   New Model: {comp['new_score']:.4f}")
            print(f"   Improvement: {comp['improvement']:.2f}%")
            print(f"   Deployed: {'✓' if comp['deploy'] else '✗'}")
    else:
        print(f"⚠️  Retraining {result['status']}: {result.get('reason', 'Unknown')}")

def get_stats():
    """Get system stats"""
    print("\n" + "="*60)
    print("📈 System Statistics")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/stats")
    stats = response.json()
    
    print(f"Total Predictions: {stats['total_predictions']}")
    print(f"Model Versions: {stats['total_versions']}")
    print(f"Auto-Retrain: {'Enabled' if stats['auto_retrain'] else 'Disabled'}")
    print(f"Drift Threshold: {stats['drift_threshold']*100:.0f}%")
    
    if stats.get('latest_version'):
        v = stats['latest_version']
        print(f"\nLatest Model:")
        print(f"  Version: {v['version']}")
        print(f"  Score: {v['score']:.4f}")
        print(f"  Samples: {v['samples']}")

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
    print("\n💡 Key Features Demonstrated:")
    print("   1. ✓ Predictions with confidence scores")
    print("   2. ✓ Automatic data logging")
    print("   3. ✓ Statistical drift detection")
    print("   4. ✓ Model validation before deployment")
    print("   5. ✓ Performance comparison (A/B testing)")
    print("   6. ✓ Model versioning with rollback")
    print("   7. ✓ Fail-safe mechanisms")
    print("\n🎯 Real-world Use Case:")
    print("   EV charging stations - demand changes during peak hours.")
    print("   System detects shift and retrains automatically.\n")
    print("📊 View Dashboard: streamlit run dashboard.py\n")

if __name__ == "__main__":
    main()
