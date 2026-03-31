import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(page_title="Preprocessing Techniques Comparison", layout="wide", initial_sidebar_state="expanded")

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
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='title'>🔧 Preprocessing Techniques Comparison</h1>", unsafe_allow_html=True)
st.markdown("Pima Indians Diabetes Dataset - AdaBoost Classifier")
st.markdown("---")

# Load data
try:
    results_df = pd.read_csv('scaler_comparison_table.csv')
except:
    st.error("Error loading comparison results")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("📊 Analysis Configuration")
    st.info("""
    **Model Configuration:**
    - Algorithm: AdaBoost Classifier
    - Estimators: 200
    - Test Size: 20%
    - Random State: 42
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📈 Accuracy Comparison")
    
    # Extract numeric accuracy values
    results_df['Accuracy_Numeric'] = results_df['Accuracy'].str.rstrip('%').astype(float)
    
    fig_accuracy = px.bar(
        results_df,
        x='Preprocessing Technique',
        y='Accuracy_Numeric',
        color='Accuracy_Numeric',
        color_continuous_scale='RdYlGn',
        labels={'Accuracy_Numeric': 'Accuracy (%)'},
        title='Accuracy by Preprocessing Technique'
    )
    fig_accuracy.update_traces(showlegend=False)
    st.plotly_chart(fig_accuracy, use_container_width=True)

with col2:
    st.subheader("📊 Detailed Metrics")
    
    # Extract metric values for radar chart
    results_df['Precision_Numeric'] = results_df['Precision'].str.rstrip('%').astype(float)
    results_df['Recall_Numeric'] = results_df['Recall'].str.rstrip('%').astype(float)
    results_df['F1-Score_Numeric'] = results_df['F1-Score'].str.rstrip('%').astype(float)
    
    # Create radar chart for first technique (MinMax)
    fig_radar = go.Figure()
    
    for idx, row in results_df.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row['Accuracy_Numeric'], row['Precision_Numeric'], 
               row['Recall_Numeric'], row['F1-Score_Numeric']],
            theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            fill='toself',
            name=row['Preprocessing Technique'],
            opacity=0.7
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title='Performance Metrics Radar Chart',
        height=500
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# Detailed comparison table
st.subheader("📋 Complete Comparison Table")
display_df = results_df[['Preprocessing Technique', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].copy()
display_df.index = range(1, len(display_df) + 1)
st.dataframe(display_df, use_container_width=True)

# Statistics
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_accuracy = results_df['Accuracy_Numeric'].mean()
    st.metric("Average Accuracy", f"{avg_accuracy:.2f}%")

with col2:
    avg_precision = results_df['Precision_Numeric'].mean()
    st.metric("Average Precision", f"{avg_precision:.2f}%")

with col3:
    avg_recall = results_df['Recall_Numeric'].mean()
    st.metric("Average Recall", f"{avg_recall:.2f}%")

with col4:
    avg_f1 = results_df['F1-Score_Numeric'].mean()
    st.metric("Average F1-Score", f"{avg_f1:.2f}%")

# Key insights
st.divider()
st.subheader("💡 Key Insights")

best_idx = results_df['Accuracy_Numeric'].idxmax()
best_technique = results_df.loc[best_idx, 'Preprocessing Technique']
best_accuracy = results_df.loc[best_idx, 'Accuracy']

col1, col2 = st.columns([1, 2])

with col1:
    st.success(f"""
    🏆 **Best Technique:** 
    {best_technique}
    
    **Accuracy:** {best_accuracy}
    """)

with col2:
    st.info("""
    **Observation:**
    All 5 preprocessing techniques achieved identical accuracy scores (74.03%).
    This suggests that for the Pima diabetes dataset with AdaBoost classifier,
    the choice of preprocessing technique has minimal impact on model performance.
    
    **Recommendation:**
    Use **MinMax Scaler** for simplicity and interpretability, as it normalizes
    features to a [0, 1] range, which is intuitive and widely understood.
    """)

st.divider()

# Preprocessing technique descriptions
st.subheader("ℹ️ Preprocessing Techniques Explained")

techniques_info = {
    'MinMax Scaler': {
        'Formula': 'X_scaled = (X - X_min) / (X_max - X_min)',
        'Range': '[0, 1]',
        'Use Case': 'Preserves zero entries, useful for bounded data',
        'Pros': 'Simple, bounds features to [0,1]',
        'Cons': 'Sensitive to outliers'
    },
    'Standard Scaler': {
        'Formula': 'X_scaled = (X - mean) / std_dev',
        'Range': '[-∞, +∞]',
        'Use Case': 'Most common, assumes normally distributed data',
        'Pros': 'Centers data around zero, variance = 1',
        'Cons': 'Sensitive to outliers'
    },
    'Robust Scaler': {
        'Formula': 'X_scaled = (X - median) / IQR',
        'Range': '[-∞, +∞]',
        'Use Case': 'Data with outliers',
        'Pros': 'Robust to outliers',
        'Cons': 'Data not scaled to fixed range'
    },
    'MaxAbs Scaler': {
        'Formula': 'X_scaled = X / max(|X|)',
        'Range': '[-1, 1]',
        'Use Case': 'Sparse data, preserves sparsity',
        'Pros': 'Preserves zeros, no shift',
        'Cons': 'Still affected by extreme values'
    },
    'Log Transformation': {
        'Formula': 'X_scaled = log(X + 1)',
        'Range': '[0, +∞]',
        'Use Case': 'Right-skewed data, exponential relationships',
        'Pros': 'Reduces skewness, handles large ranges',
        'Cons': 'Cannot handle negative values (without shift)'
    }
}

for technique, info in techniques_info.items():
    with st.expander(f"📌 {technique}"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Formula:** `{info['Formula']}`")
            st.write(f"**Output Range:** {info['Range']}")
        with col2:
            st.write(f"**Use Case:** {info['Use Case']}")
        st.write(f"**Pros:** {info['Pros']}")
        st.write(f"**Cons:** {info['Cons']}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; padding: 20px; color: gray;'>
    <small>📊 Data-driven preprocessing comparison for Pima Indians Diabetes prediction | Model: AdaBoost Classifier</small>
</div>
""", unsafe_allow_html=True)
