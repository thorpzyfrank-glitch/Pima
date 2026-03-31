import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Logistic Regression Model",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Add custom styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 44px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 30px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .model-name {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .accuracy-score {
        font-size: 56px;
        font-weight: bold;
        margin: 20px 0;
        color: #ffd700;
    }
    .description {
        font-size: 16px;
        margin-top: 10px;
        opacity: 0.95;
    }
    .stat-box {
        background: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">🤖 Logistic Regression Model</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Pima Indians Diabetes Classification</div>', unsafe_allow_html=True)

# Load the trained logistic regression model
with open('logistic_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

# Load and prepare data for metrics
df = pd.read_csv('Pima.csv')
X = df.drop('diabetes_class', axis=1)
y = df['diabetes_class']

# Calculate metrics
y_pred = lr_model.predict(X)
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

# Display main metric container
st.markdown(f"""
    <div class="metric-container">
        <div class="model-name">📊 Logistic Regression</div>
        <div class="description">Binary Classification Model</div>
        <div class="accuracy-score">{accuracy*100:.2f}%</div>
        <div class="description">Model Accuracy</div>
    </div>
""", unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Metrics", "📋 Model Details", "🔍 Predictions", "📊 Visualization"])

with tab1:
    st.subheader("Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}", f"{accuracy*100:.2f}%")
    
    with col2:
        st.metric("Precision", f"{precision:.4f}", f"{precision*100:.2f}%")
    
    with col3:
        st.metric("Recall", f"{recall:.4f}", f"{recall*100:.2f}%")
    
    with col4:
        st.metric("F1-Score", f"{f1:.4f}", f"{f1*100:.2f}%")
    
    st.divider()
    
    # Detailed metrics
    st.write("### Performance Summary")
    st.markdown(f"""
    <div class="stat-box">
        <strong>Accuracy:</strong> {accuracy:.4f} ({accuracy*100:.2f}%)<br>
        Model correctly predicts diabetes status {accuracy*100:.2f}% of the time
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stat-box">
        <strong>Precision:</strong> {precision:.4f} ({precision*100:.2f}%)<br>
        Of positive predictions, {precision*100:.2f}% are correct (True Positive Rate)
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stat-box">
        <strong>Recall:</strong> {recall:.4f} ({recall*100:.2f}%)<br>
        Model identifies {recall*100:.2f}% of actual positive cases
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stat-box">
        <strong>F1-Score:</strong> {f1:.4f} ({f1*100:.2f}%)<br>
        Harmonic mean of precision and recall
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.subheader("Model Configuration & Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Dataset Information")
        st.write(f"**Total Samples:** {len(df)}")
        st.write(f"**Number of Features:** {len(X.columns)}")
        st.write(f"**Target Variable:** Diabetes Class (Binary)")
        st.write(f"**Positive Cases:** {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")
        st.write(f"**Negative Cases:** {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
    
    with col2:
        st.write("### Model Algorithm")
        st.write("**Algorithm:** Logistic Regression (Linear Model)")
        st.write("**Solver:** lbfgs")
        st.write("**Max Iterations:** 1000")
        st.write("**Random State:** 42")
        st.write("**Regularization:** L2 (Default)")
    
    st.divider()
    
    st.write("### Features Used for Classification")
    features = X.columns.tolist()
    for i, feature in enumerate(features, 1):
        st.write(f"{i}. **{feature}**")

with tab3:
    st.subheader("Make Predictions")
    
    st.write("Enter patient health metrics to predict diabetes risk:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        preg_count = st.slider("Pregnancies", 0, 15, 2)
    with col2:
        glucose = st.slider("Glucose Concentration", 0, 200, 100)
    with col3:
        diastolic_bp = st.slider("Diastolic BP", 0, 120, 70)
    with col4:
        triceps = st.slider("Triceps Skin Fold", 0, 100, 20)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        insulin = st.slider("2-hr Serum Insulin", 0, 850, 80)
    with col2:
        bmi = st.slider("BMI", 0.0, 65.0, 25.0)
    with col3:
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    with col4:
        age = st.slider("Age", 15, 85, 30)
    
    # Make prediction
    input_data = np.array([[preg_count, glucose, diastolic_bp, triceps, insulin, bmi, dpf, age]])
    prediction = lr_model.predict(input_data)[0]
    probability = lr_model.predict_proba(input_data)[0]
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Prediction Result")
        if prediction == 1:
            st.error("⚠️ **POSITIVE** - Diabetes Risk Detected", icon="⚠️")
        else:
            st.success("✅ **NEGATIVE** - No Diabetes Detected", icon="✅")
    
    with col2:
        st.write("### Probability")
        st.metric("Negative (0)", f"{probability[0]*100:.2f}%")
        st.metric("Positive (1)", f"{probability[1]*100:.2f}%")

with tab4:
    st.subheader("Model Visualization")
    
    # Confusion Matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'],
                ax=ax, annot_kws={'size': 16})
    ax.set_ylabel('Actual', fontweight='bold', fontsize=12)
    ax.set_xlabel('Predicted', fontweight='bold', fontsize=12)
    ax.set_title('Confusion Matrix - Logistic Regression', fontweight='bold', fontsize=14)
    st.pyplot(fig)
    
    # Metrics visualization
    st.write("### Key Metrics Comparison")
    
    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [accuracy, precision, recall, f1]
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics_df['Metric'], metrics_df['Score']*100, 
                   color=['#667eea', '#764ba2', '#8fd3f4', '#84fab0'], alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Score (%)', fontweight='bold', fontsize=12)
    ax.set_title('Model Performance Metrics', fontweight='bold', fontsize=14)
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)

# Sidebar
st.sidebar.title("📋 Model Information")
st.sidebar.write(f"**Model Name:** Logistic Regression")
st.sidebar.write(f"**Accuracy:** {accuracy:.4f} ({accuracy*100:.2f}%)")
st.sidebar.write(f"**Precision:** {precision:.4f}")
st.sidebar.write(f"**Recall:** {recall:.4f}")
st.sidebar.write(f"**F1-Score:** {f1:.4f}")

st.sidebar.divider()

st.sidebar.title("🎯 Model Type")
st.sidebar.write("**Algorithm:** Logistic Regression")
st.sidebar.write("**Task:** Binary Classification")
st.sidebar.write("**Target:** Diabetes Prediction")

st.sidebar.divider()

st.sidebar.title("📊 Dataset")
st.sidebar.write(f"**Samples:** {len(df)}")
st.sidebar.write(f"**Features:** {len(X.columns)}")
st.sidebar.write(f"**Positive Cases:** {(y == 1).sum()}")
st.sidebar.write(f"**Negative Cases:** {(y == 0).sum()}")

# Footer
st.divider()

st.markdown("""
---
### 🏥 Logistic Regression Classification Model

**Model Status:** ✅ Successfully Trained & Deployed

**Summary:**
- **Model Name:** Logistic Regression
- **Accuracy Score:** {:.4f} ({:.2f}%)
- **Precision:** {:.4f}
- **Recall:** {:.4f}
- **F1-Score:** {:.4f}

**Framework:** scikit-learn, Streamlit  
**Last Updated:** 2026-03-25  
**Dataset:** Pima Indians Diabetes

---
""".format(accuracy, accuracy*100, precision, recall, f1))
