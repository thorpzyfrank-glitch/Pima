import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="Imbalance Solutions Comparison", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
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
st.markdown("<h1 class='title'>⚖️ Class Imbalance Solutions Comparison</h1>", unsafe_allow_html=True)
st.markdown("---")

# Load data
results_df = pd.read_csv('imbalance_solutions_comparison.csv')

# Sidebar
with st.sidebar:
    st.header("📊 Dataset Information")
    st.info("""
    **Pima Indians Diabetes Dataset**
    - Total Samples: 768
    - Features: 8
    - Diabetes Cases: 268 (34.9%)
    - Non-Diabetes: 500 (65.1%)
    - Imbalance Ratio: 1.87:1
    """)
    
    st.divider()
    st.subheader("🔍 Resampling Techniques")
    st.markdown("""
    **1. No Resampling (Baseline)**
    - Original imbalanced data
    - Training samples: 614
    
    **2. SMOTE**
    - Creates synthetic minority samples
    - Training samples: 800 (balanced)
    
    **3. Random Over Sampler**
    - Duplicates minority samples randomly
    - Training samples: 800 (balanced)
    """)

# Create overview metrics
col1, col2, col3 = st.columns(3)

with col1:
    avg_accuracy_baseline = results_df[results_df['Technique'] == 'No Resampling']['Accuracy'].mean()
    st.metric("Baseline Accuracy", f"{avg_accuracy_baseline:.2%}")

with col2:
    avg_accuracy_smote = results_df[results_df['Technique'] == 'SMOTE']['Accuracy'].mean()
    st.metric("SMOTE Accuracy", f"{avg_accuracy_smote:.2%}")

with col3:
    avg_accuracy_ros = results_df[results_df['Technique'] == 'Random Over Sampler']['Accuracy'].mean()
    st.metric("Random Over Sampler Accuracy", f"{avg_accuracy_ros:.2%}")

st.divider()

# Tabs for different visualizations
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Accuracy Comparison", "🎯 Detailed Metrics", "📉 Model Comparison", "📋 Data Table"])

# ========== TAB 1: OVERVIEW ==========
with tab1:
    st.subheader("Summary by Resampling Technique")
    
    summary = results_df.groupby('Technique')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].mean().round(4)
    summary = summary.reset_index()
    summary = summary.sort_values('Accuracy', ascending=False)
    
    st.dataframe(summary, use_container_width=True)
    
    st.subheader("Class Distribution Impact")
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.info("""
        **Original Dataset**
        - Negative (Non-Diabetes): 500 (65.1%)
        - Positive (Diabetes): 268 (34.9%)
        - Imbalance Ratio: 1.87:1
        """)
    
    with col_right:
        st.success("""
        **After Resampling**
        - Both SMOTE and ROS created balanced data
        - Negative: 400 (50%)
        - Positive: 400 (50%)
        - Imbalance Ratio: 1:1
        """)

# ========== TAB 2: ACCURACY COMPARISON ==========
with tab2:
    st.subheader("Accuracy Scores by Technique and Model")
    
    # Accuracy comparison by technique
    accuracy_pivot = results_df.pivot(index='Model', columns='Technique', values='Accuracy')
    
    fig = go.Figure()
    
    for technique in ['No Resampling', 'SMOTE', 'Random Over Sampler']:
        if technique in accuracy_pivot.columns:
            fig.add_trace(go.Bar(
                x=accuracy_pivot.index,
                y=accuracy_pivot[technique],
                name=technique,
                text=[f'{val:.2%}' for val in accuracy_pivot[technique]],
                textposition='auto'
            ))
    
    fig.update_layout(
        title="Accuracy Comparison by Model",
        xaxis_title="Model",
        yaxis_title="Accuracy",
        barmode='group',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Best model per technique
    col1, col2, col3 = st.columns(3)
    
    techniques = ['No Resampling', 'SMOTE', 'Random Over Sampler']
    cols = [col1, col2, col3]
    
    for technique, col in zip(techniques, cols):
        best_model = results_df[results_df['Technique'] == technique].loc[results_df[results_df['Technique'] == technique]['Accuracy'].idxmax()]
        with col:
            st.success(f"""
            **{technique}**
            - Best Model: {best_model['Model']}
            - Accuracy: {best_model['Accuracy']:.2%}
            """)

# ========== TAB 3: DETAILED METRICS ==========
with tab3:
    st.subheader("Comprehensive Metrics Comparison")
    
    # Create heatmap for all metrics
    metrics_data = results_df.groupby('Technique')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].mean()
    
    fig = go.Figure(data=go.Heatmap(
        z=metrics_data.values,
        x=metrics_data.columns,
        y=metrics_data.index,
        colorscale='Viridis',
        text=metrics_data.values.round(4),
        texttemplate='%{text:.4f}',
        textfont={"size": 12},
        colorbar=dict(title="Score")
    ))
    
    fig.update_layout(
        title="Performance Metrics Heatmap",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Precision vs Recall comparison
    st.subheader("Precision vs Recall Trade-off")
    
    fig = go.Figure()
    
    for technique in results_df['Technique'].unique():
        data = results_df[results_df['Technique'] == technique]
        fig.add_trace(go.Scatter(
            x=data['Recall'],
            y=data['Precision'],
            mode='markers+text',
            name=technique,
            text=data['Model'],
            textposition='top center',
            marker=dict(size=10)
        ))
    
    fig.update_layout(
        title="Precision vs Recall by Technique",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=500,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ========== TAB 4: MODEL COMPARISON ==========
with tab4:
    st.subheader("Select Model for Detailed Comparison")
    
    selected_model = st.selectbox("Choose a model:", results_df['Model'].unique())
    
    model_data = results_df[results_df['Model'] == selected_model]
    
    # Create radar chart for the selected model
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    fig = go.Figure()
    
    for _, row in model_data.iterrows():
        values = [row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score'], row['AUC-ROC']]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=row['Technique']
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"Performance Radar: {selected_model}",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table for selected model
    st.subheader(f"Detailed Metrics for {selected_model}")
    st.dataframe(model_data[['Technique', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].round(4), use_container_width=True)

# ========== TAB 5: DATA TABLE ==========
with tab5:
    st.subheader("Complete Results Table")
    
    st.dataframe(
        results_df.sort_values(['Technique', 'Accuracy'], ascending=[True, False]).round(4),
        use_container_width=True
    )
    
    # Download CSV
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="imbalance_solutions_comparison.csv",
        mime="text/csv"
    )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; padding: 20px; color: gray;'>
    <small>
    💡 **Key Findings:**
    - SMOTE and Random Over Sampler both create balanced datasets but with different approaches
    - SMOTE creates synthetic samples while ROS duplicates existing minority samples
    - All three techniques have different impact on model recall (sensitivity to positive class)
    - For medical diagnoses, recall is often more important than accuracy
    </small>
</div>
""", unsafe_allow_html=True)
