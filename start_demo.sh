#!/bin/bash

echo "============================================================"
echo "🚀 MLOps Auto-Retrain System - Quick Start"
echo "============================================================"

# Check if model exists
if [ ! -f "models/model.pkl" ]; then
    echo ""
    echo "📦 Training initial model..."
    python ml_model/train.py
    echo ""
fi

echo "✅ Setup complete!"
echo ""
echo "Run these commands in separate terminals:"
echo ""
echo "  Terminal 1: python run.py"
echo "  Terminal 2: python test_system.py"
echo "  Terminal 3: streamlit run dashboard.py"
echo ""
echo "============================================================"
