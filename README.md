# MLOps Auto-Retrain System

An automated machine learning operations system that continuously monitors model performance, detects data drift, and retrains models without manual intervention.

## Problem Statement

Machine learning models in production lose accuracy over time due to data drift. Traditional systems require manual monitoring and retraining, making them costly and unreliable for real-world applications.

## Solution

A lightweight automated MLOps pipeline that:
- Serves predictions via REST API
- Monitors incoming data continuously
- Detects drift using statistical analysis
- Validates and retrains models automatically
- Maintains version control with rollback capability

## Architecture

```
Input → Prediction → Data Logging → Drift Detection → Model Retraining → Deployment
```

## Features

- **Automated Drift Detection**: Statistical monitoring of data distribution changes
- **Smart Retraining**: Performance validation before deployment
- **Model Versioning**: Full version history with rollback support
- **Confidence Scores**: Prediction reliability estimation
- **Model Upload**: Support for custom trained models (.pkl)
- **Multi-Algorithm Support**: Random Forest, Linear Regression, Gradient Boosting
- **Real-time Dashboard**: Clean monitoring interface
- **Fail-safe Mechanisms**: Automatic rollback on training failures

## Installation

```bash
# Clone repository
git clone https://github.com/MohamedAshiq09/micro_engine_orchestra_mlops.git
cd micro_engine_orchestra_mlops

# Install dependencies
pip install -r requirements.txt

# Train initial model
python ml_model/train.py
```

## Usage

### Start API Server
```bash
python run.py
```
API available at `http://localhost:8000`

### Launch Dashboard
```bash
streamlit run dashboard.py
```
Dashboard available at `http://localhost:8501`

### Run Demo
```bash
python test_system.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Make predictions with confidence scores |
| `/check_drift` | GET | Check for data drift |
| `/retrain` | POST | Trigger model retraining |
| `/upload_model` | POST | Upload custom model (.pkl) |
| `/train_new_model` | POST | Train new model from scratch |
| `/versions` | GET | List all model versions |
| `/stats` | GET | System statistics |
| `/rollback/{version}` | POST | Rollback to specific version |

## Project Structure

```
mlops-auto-retrain/
├── mlops_system/          # Core system
│   ├── api.py            # FastAPI endpoints
│   ├── model_manager.py  # Model training/loading
│   ├── drift_detector.py # Drift detection logic
│   ├── data_store.py     # Data logging
│   ├── version_manager.py # Version control
│   └── visualizer.py     # Drift visualization
├── ml_model/             # Sample ML model
│   └── train.py         # Initial model training
├── models/              # Saved models
├── logs/                # Prediction logs
├── dashboard.py         # Streamlit UI
├── test_system.py       # Automated demo
├── run.py              # Server launcher
└── requirements.txt    # Dependencies
```

## Dashboard Features

### Main View
- Total predictions count
- Current model score
- Drift detection status
- Model version count

### Sidebar
- **Upload Model**: Deploy custom .pkl models
- **Train New**: Train models with different algorithms

### Visualizations
- Feature drift analysis (bar charts)
- Model version history
- Performance metrics

## Demo Workflow

1. **Normal Predictions**: System serves predictions and logs data
2. **Drift Detection**: Monitors for distribution changes
3. **Simulated Drift**: Introduces shifted data patterns
4. **Automatic Retraining**: Validates and deploys improved model
5. **Performance Comparison**: Shows old vs new model metrics

## Real-World Use Case

**EV Charging Stations**: Demand patterns change during peak hours and seasons. The system detects these shifts and retrains automatically, maintaining prediction accuracy without manual intervention.

## Key Metrics

- **Drift Detection**: Statistical distance (mean/std change > 30%)
- **Model Validation**: Train/validation split with R² scoring
- **Deployment Criteria**: New model must outperform current model
- **Confidence Calculation**: Distance-based reliability estimation

## Technical Stack

- **Backend**: FastAPI, Python 3.14+
- **ML**: scikit-learn (Random Forest, Linear Regression, Gradient Boosting)
- **Frontend**: Streamlit
- **Visualization**: Plotly, Matplotlib
- **Data Processing**: NumPy

## Configuration

Edit `mlops_system/api.py`:
```python
DRIFT_THRESHOLD = 0.3      # 30% change triggers drift
MIN_IMPROVEMENT = 0.0      # Minimum improvement to deploy
AUTO_RETRAIN = True        # Enable automatic retraining
```

## Future Enhancements

- Custom training pipeline integration
- Multi-model ensemble support
- Advanced drift detection algorithms
- Cloud deployment support
- Real-time alerting system

## License

MIT License

## Repository

https://github.com/MohamedAshiq09/micro_engine_orchestra_mlops

---

**Note**: This system demonstrates core MLOps principles including continuous monitoring, automated retraining, and production-grade model management.
