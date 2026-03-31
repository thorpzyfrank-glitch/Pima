import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(page_title="Pima Diabetes Predictor", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .title {
        color: #1f77b4;
        text-align: center;
    }
    .metric {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='title'>🏥 Pima Indians Diabetes Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

# Load model and scaler
try:
    model = joblib.load('pima_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
except:
    st.error("⚠️ Model files not found. Please ensure 'pima_model.pkl', 'scaler.pkl', and 'feature_names.pkl' are in the same directory.")
    st.stop()

# Sidebar - Information
with st.sidebar:
    st.header("📊 Model Information")
    st.info("""
    **Gradient Boosting Classifier**
    - Trained on Pima Indians Diabetes Dataset
    - 8 input features
    - High accuracy prediction model
    """)
    
    st.divider()
    st.subheader("🎯 Feature Ranges")
    st.text("Pregnancies: 0-17")
    st.text("Glucose: 44-199")
    st.text("BP: 24-122")
    st.text("Skin Fold: 0-99")
    st.text("Insulin: 0-846")
    st.text("BMI: 18.2-67.1")
    st.text("Diabetes Pedigree: 0.08-2.42")
    st.text("Age: 21-81")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter Patient Information")
    
    # Create input fields
    col_left, col_right = st.columns(2)
    
    with col_left:
        preg_count = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose Concentration (mg/dL)", min_value=0, max_value=300, value=120)
        diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=0, max_value=150, value=70)
        triceps = st.number_input("Triceps Skin Fold Thickness (mm)", min_value=0, max_value=100, value=20)
    
    with col_right:
        insulin = st.number_input("2-Hour Serum Insulin (mu U/ml)", min_value=0, max_value=900, value=100)
        bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=25.0)
        diabetes_pedi = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=35)

with col2:
    st.subheader("🔮 Prediction")
    
    if st.button("🚀 Predict", use_container_width=True, type="primary"):
        # Prepare input data
        input_data = np.array([[preg_count, glucose, diastolic_bp, triceps, insulin, bmi, diabetes_pedi, age]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Display results
        st.divider()
        
        if prediction == 1:
            st.error(f"⚠️ **Positive for Diabetes**")
            st.metric("Probability", f"{probability[1]*100:.2f}%")
        else:
            st.success(f"✅ **Negative for Diabetes**")
            st.metric("Probability", f"{probability[0]*100:.2f}%")
        
        # Risk assessment
        st.divider()
        st.subheader("📈 Risk Assessment")
        risk_score = probability[1]
        
        col_risk1, col_risk2 = st.columns(2)
        with col_risk1:
            st.metric("Diabetes Risk Score", f"{risk_score:.2%}")
        with col_risk2:
            st.metric("Non-Diabetes Probability", f"{probability[0]:.2%}")
        
        # Recommendation
        st.divider()
        if risk_score > 0.7:
            st.warning("🚨 **High Risk** - Please consult with a healthcare professional immediately")
        elif risk_score > 0.4:
            st.info("⚠️ **Moderate Risk** - Regular check-ups are recommended")
        else:
            st.success("✅ **Low Risk** - Continue maintaining a healthy lifestyle")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; padding: 20px; color: gray;'>
    <small>💡 Disclaimer: This tool is for educational purposes only and should not be used as a substitute for professional medical advice.</small>
</div>
""", unsafe_allow_html=True)
