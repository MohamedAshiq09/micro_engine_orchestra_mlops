#!/usr/bin/env python3
import streamlit as st
import requests
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import time
import pickle
import io

st.set_page_config(page_title="MLOps Monitor", layout="wide", initial_sidebar_state="collapsed")

BASE_URL = "http://localhost:8000"

# Clean CSS with proper contrast
st.markdown("""
<style>
    .main {background-color: #f5f7fa;}
    .stMetric {
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 8px;
        border: 1px solid #e1e4e8;
    }
    .stMetric label {
        color: #586069 !important;
        font-size: 14px !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #24292e !important;
        font-size: 28px !important;
        font-weight: 600 !important;
    }
    h1 {color: #24292e; font-weight: 400;}
    h3 {color: #24292e; font-weight: 400;}
    .stButton button {
        background-color: #0366d6;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 6px;
        font-weight: 500;
    }
    .stButton button:hover {
        background-color: #0256c7;
    }
    .upload-section {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e1e4e8;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("MLOps Monitor")

# Sidebar for model management
with st.sidebar:
    st.header("Model Management")
    
    tab1, tab2 = st.tabs(["Upload Model", "Train New"])
    
    with tab1:
        st.markdown("### Upload Model")
        uploaded_file = st.file_uploader("Upload .pkl model file", type=['pkl'])
        
        if uploaded_file is not None:
            if st.button("Deploy Uploaded Model", key="upload_btn"):
                try:
                    files = {'file': uploaded_file.getvalue()}
                    response = requests.post(f"{BASE_URL}/upload_model", files={'file': uploaded_file})
                    
                    if response.status_code == 200:
                        st.success("✅ Model uploaded and deployed!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ Upload failed: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    with tab2:
        st.markdown("### Train New Model")
        
        model_type = st.selectbox(
            "Model Type",
            ["Random Forest", "Linear Regression", "Gradient Boosting"]
        )
        
        n_samples = st.slider("Training Samples", 100, 2000, 1000, 100)
        
        if st.button("Train Model", key="train_btn"):
            with st.spinner("Training..."):
                try:
                    response = requests.post(
                        f"{BASE_URL}/train_new_model",
                        json={"model_type": model_type, "n_samples": n_samples}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ Model trained! Score: {result['score']:.4f}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Training failed")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# Check server
try:
    response = requests.get(BASE_URL, timeout=2)
    server_status = "🟢 Online"
except:
    st.error("⚠️ Server offline. Run: python run.py")
    st.stop()

# Main metrics
col1, col2, col3, col4 = st.columns(4)

try:
    stats = requests.get(f"{BASE_URL}/stats").json()
    drift_check = requests.get(f"{BASE_URL}/check_drift").json()
    
    with col1:
        st.metric("Predictions", stats['total_predictions'], delta=None)
    
    with col2:
        latest = stats.get('latest_version')
        score = latest['score'] if latest else 0
        st.metric("Model Score", f"{score:.4f}", delta=None)
    
    with col3:
        drift_status = "Detected" if drift_check.get('drift_detected') else "Normal"
        drift_delta = "⚠️" if drift_check.get('drift_detected') else "✓"
        st.metric("Drift Status", drift_status, delta=drift_delta)
    
    with col4:
        st.metric("Versions", stats['total_versions'], delta=None)
    
    st.divider()
    
    # Drift visualization
    if drift_check.get('drift_detected'):
        st.subheader("Drift Analysis")
        
        drift_info = drift_check.get('drift_info', {})
        features = drift_info.get('features', {})
        
        if features:
            feature_names = list(features.keys())
            mean_changes = [features[f]['mean_change'] * 100 for f in feature_names]
            
            fig = go.Figure(data=[
                go.Bar(x=feature_names, y=mean_changes, marker_color='#e74c3c', text=mean_changes, texttemplate='%{text:.1f}%', textposition='outside')
            ])
            fig.update_layout(
                title="Feature Drift (%)",
                xaxis_title="Features",
                yaxis_title="Change %",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(color='#24292e')
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Model versions
    st.subheader("Model Versions")
    
    versions_data = requests.get(f"{BASE_URL}/versions").json()
    versions = versions_data.get('versions', [])
    
    if versions:
        st.markdown("""
        <style>
        .version-row {
            background-color: #ffffff;
            padding: 12px;
            margin: 8px 0;
            border-radius: 6px;
            border: 1px solid #e1e4e8;
            color: #24292e;
        }
        </style>
        """, unsafe_allow_html=True)
        
        for v in reversed(versions[-5:]):  # Show last 5
            st.markdown(f"""
            <div class="version-row">
                <strong>v{v['version']}</strong> &nbsp;&nbsp;|&nbsp;&nbsp; 
                Score: {v['score']:.4f} &nbsp;&nbsp;|&nbsp;&nbsp; 
                Samples: {v['samples']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No versions yet")
    
    # Actions
    st.divider()
    col_x, col_y = st.columns(2)
    
    with col_x:
        if st.button("🔄 Retrain Model", use_container_width=True):
            with st.spinner("Retraining..."):
                result = requests.post(f"{BASE_URL}/retrain").json()
                if result['status'] == 'success':
                    st.success(f"✅ Retrained! Score: {result['val_score']:.4f}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.warning(f"⚠️ {result.get('reason', 'Failed')}")
    
    with col_y:
        if st.button("📊 Check Drift", use_container_width=True):
            st.rerun()

except Exception as e:
    st.error(f"Error: {e}")
