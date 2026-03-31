import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="ML Model Comparison", layout="wide", initial_sidebar_state="expanded")

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
st.markdown("<h1 class='title'>📊 10 Classification Algorithms Comparison</h1>", unsafe_allow_html=True)
st.markdown("Pima Indians Diabetes Dataset Analysis")
st.markdown("---")

# Load comparison data
comparison_data = {
    'Model': ['AdaBoost', 'Random Forest', 'Support Vector Machine', 'Gradient Boosting', 
              'XGBoost', 'Decision Tree', 'Logistic Regression', 'Naive Bayes', 
              'K-Nearest Neighbors', 'Neural Network (MLP)'],
    'Accuracy': [0.7922, 0.7597, 0.7532, 0.7532, 0.7338, 0.7208, 0.7143, 0.7078, 0.7013, 0.7013],
    'Precision': [0.7200, 0.6809, 0.6600, 0.6667, 0.6226, 0.6341, 0.6087, 0.5738, 0.5833, 0.5625],
    'Recall': [0.6667, 0.5926, 0.6111, 0.5926, 0.6111, 0.4815, 0.5185, 0.6481, 0.5185, 0.6667],
    'F1-Score': [0.6923, 0.6337, 0.6346, 0.6275, 0.6168, 0.5474, 0.5600, 0.6087, 0.5490, 0.6102]
}

df = pd.DataFrame(comparison_data)

# Main metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    best_model = df.loc[df['Accuracy'].idxmax()]
    st.metric("🏆 Best Model", best_model['Model'], f"{best_model['Accuracy']:.2%}")

with col2:
    st.metric("📈 Average Accuracy", f"{df['Accuracy'].mean():.2%}")

with col3:
    st.metric("📊 Top Accuracy Range", f"{df['Accuracy'].max() - df['Accuracy'].min():.2%}")

with col4:
    st.metric("🎯 Models Evaluated", len(df))

st.markdown("---")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["Accuracy Comparison", "Detailed Metrics", "Performance Radar", "Detailed Table"])

# Tab 1: Accuracy Comparison
with tab1:
    st.subheader("Accuracy Scores by Model")
    
    # Bar chart
    fig_accuracy = px.bar(df.sort_values('Accuracy', ascending=True), 
                          x='Accuracy', 
                          y='Model',
                          orientation='h',
                          color='Accuracy',
                          color_continuous_scale='RdYlGn',
                          text='Accuracy',
                          labels={'Accuracy': 'Accuracy Score'})
    
    fig_accuracy.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig_accuracy.layout.height = 500
    st.plotly_chart(fig_accuracy, use_container_width=True)
    
    # Summary text
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Best Performer:** {df.loc[df['Accuracy'].idxmax(), 'Model']} ({df['Accuracy'].max():.2%})")
    with col2:
        st.warning(f"**Needs Improvement:** {df.loc[df['Accuracy'].idxmin(), 'Model']} ({df['Accuracy'].min():.2%})")

# Tab 2: Detailed Metrics
with tab2:
    st.subheader("All Performance Metrics")
    
    fig_metrics = go.Figure()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    for metric in metrics:
        fig_metrics.add_trace(go.Bar(
            name=metric,
            x=df['Model'],
            y=df[metric],
            text=[f'{val:.2%}' for val in df[metric]],
            textposition='auto',
        ))
    
    fig_metrics.update_layout(
        barmode='group',
        height=500,
        xaxis_tickangle=-45,
        yaxis_title="Score",
        hovermode='x unified'
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

# Tab 3: Performance Radar
with tab3:
    st.subheader("Model Performance Comparison (Top 5)")
    
    top_5_models = df.nlargest(5, 'Accuracy')
    
    fig_radar = go.Figure()
    
    for idx, row in top_5_models.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']],
            theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            fill='toself',
            name=row['Model']
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=600,
        hovermode='closest'
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# Tab 4: Detailed Table
with tab4:
    st.subheader("Complete Model Comparison Table")
    
    # Format the dataframe for display
    display_df = df.copy()
    display_df['Accuracy'] = display_df['Accuracy'].apply(lambda x: f"{x:.2%}")
    display_df['Precision'] = display_df['Precision'].apply(lambda x: f"{x:.2%}")
    display_df['Recall'] = display_df['Recall'].apply(lambda x: f"{x:.2%}")
    display_df['F1-Score'] = display_df['F1-Score'].apply(lambda x: f"{x:.2%}")
    
    st.dataframe(display_df, use_container_width=True)
    
    # Download CSV
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="model_comparison_results.csv",
        mime="text/csv"
    )

st.markdown("---")

# Footer with insights
st.subheader("🎯 Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    **Top 3 Performers:**
    1. 🥇 AdaBoost - 79.22%
    2. 🥈 Random Forest - 75.97%
    3. 🥉 SVM - 75.32%
    """)

with col2:
    st.info("""
    **Model Categories:**
    - **Ensemble Methods:** AdaBoost, Random Forest, GB, XGBoost
    - **Distance-based:** KNN, SVM
    - **Linear Models:** Logistic Regression
    - **Probabilistic:** Naive Bayes, Decision Tree
    - **Neural Networks:** MLP
    """)

st.markdown("""
---
**Dataset Information:**
- Total Samples: 768
- Training Set: 614 (80%)
- Test Set: 154 (20%)
- Features: 8
- Target: Diabetes Classification (Binary)

**Conclusion:** AdaBoost emerges as the best-performing model with 79.22% accuracy, combining multiple weak learners for robust predictions.
""")
