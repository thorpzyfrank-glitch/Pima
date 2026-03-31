import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Comprehensive Model & Scaler Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 20px;
        margin-bottom: 30px;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">🧠 Complete Model Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">14 Models × 5 Scalers = 70 Combinations</div>', unsafe_allow_html=True)

# Load results
with open('comprehensive_results.pkl', 'rb') as f:
    comprehensive_results = pickle.load(f)

with open('best_configurations.pkl', 'rb') as f:
    configurations_df = pickle.load(f)

# Create main comparison dataframe
result_df = pd.DataFrame(comprehensive_results)
all_models = [
    'Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM',
    'K-Nearest Neighbors', 'Naive Bayes', 'Gradient Boosting',
    'AdaBoost', 'Voting Classifier', 
    'ANN-Small', 'ANN-Medium', 'ANN-Large', 'ANN-Deep', 'ANN-Wide'
]
result_df = result_df.reindex(all_models)

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📋 Main Table", "🧠 Neural Networks", "📈 Visualizations", "🏆 Top Combinations", "📊 Statistics", "💡 Insights"])

with tab1:
    st.subheader("Complete Accuracy Comparison - All 14 Models × 5 Scalers")
    
    # Display as numbers with percentages
    display_df = result_df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}\n({x*100:.2f}%)")
    
    st.dataframe(display_df, use_container_width=True)
    
    # Download CSV
    csv_data = result_df.to_csv()
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name="comprehensive_results.csv",
        mime="text/csv"
    )

with tab2:
    st.subheader("🤖 Artificial Neural Network Architectures")
    
    ann_models = ['ANN-Small', 'ANN-Medium', 'ANN-Large', 'ANN-Deep', 'ANN-Wide']
    ann_df = result_df.loc[ann_models]
    
    # Display ANN results
    st.write("**Architecture Details:**")
    st.write("""
    - **ANN-Small:** (50,) - Single hidden layer with 50 neurons
    - **ANN-Medium:** (100, 50) - Two layers: 100 and 50 neurons
    - **ANN-Large:** (256, 128, 64) - Three layers: 256, 128, and 64 neurons
    - **ANN-Deep:** (200, 150, 100, 50) - Four layers: gradually decreasing
    - **ANN-Wide:** (500,) - Single very wide layer with 500 neurons
    """)
    
    st.divider()
    
    # ANN performance comparison table
    st.subheader("ANN Performance Across Scalers")
    
    display_ann_df = ann_df.copy()
    for col in display_ann_df.columns:
        display_ann_df[col] = display_ann_df[col].apply(lambda x: f"{x:.4f} ({x*100:.2f}%)")
    
    st.dataframe(display_ann_df, use_container_width=True)
    
    st.divider()
    
    # ANN statistics
    st.subheader("ANN Performance Statistics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    for idx, ann_name in enumerate(ann_models):
        col = [col1, col2, col3, col4, col5][idx]
        scores = result_df.loc[ann_name]
        with col:
            st.write(f"**{ann_name}**")
            st.metric("Avg", f"{scores.mean()*100:.2f}%")
            st.metric("Best", f"{scores.max()*100:.2f}%")
            st.metric("Worst", f"{scores.min()*100:.2f}%")

with tab3:
    st.subheader("Heatmap - Complete Model Performance")
    
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(result_df, annot=True, fmt='.3f', cmap='RdYlGn', cbar_kws={'label': 'Accuracy'},
                ax=ax, linewidths=0.5, vmin=0.62, vmax=0.79)
    ax.set_title('Accuracy Scores Heatmap - All Models × Scalers', fontweight='bold', fontsize=14, pad=20)
    ax.set_xlabel('Scaler Type', fontweight='bold')
    ax.set_ylabel('Classification Model', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.divider()
    
    st.subheader("Model Category Comparison")
    
    traditional_models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM',
                         'K-Nearest Neighbors', 'Naive Bayes', 'Gradient Boosting',
                         'AdaBoost', 'Voting Classifier']
    
    traditional_avg = result_df.loc[traditional_models].mean()
    ann_avg = result_df.loc[ann_models].mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(result_df.columns))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, traditional_avg.values*100, width, label='Traditional Models', color='#667eea', alpha=0.8)
    bars2 = ax.bar(x + width/2, ann_avg.values*100, width, label='Neural Networks', color='#764ba2', alpha=0.8)
    
    ax.set_xlabel('Scaler Type', fontweight='bold', fontsize=12)
    ax.set_ylabel('Average Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('Traditional Models vs Neural Networks', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(result_df.columns, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)

with tab4:
    st.subheader("🏆 Top 20 Best Model-Scaler Combinations")
    
    top_configs = configurations_df.head(20).reset_index(drop=True)
    top_configs['Rank'] = range(1, len(top_configs) + 1)
    
    display_configs = top_configs[['Rank', 'Model', 'Scaler', 'Model_Type', 'Accuracy']].copy()
    display_configs['Accuracy'] = display_configs['Accuracy'].apply(lambda x: f"{x:.4f} ({x*100:.2f}%)")
    display_configs['Model_Type'] = display_configs['Model_Type'].apply(lambda x: '🤖' if x == 'Neural Network' else '📊')
    
    st.dataframe(display_configs, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Visualize top 10
    st.subheader("Visualization of Top 10 Combinations")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_10 = configurations_df.head(10)
    y_labels = [f"{row['Model'][:20]} + {row['Scaler'][:15]}" for _, row in top_10.iterrows()]
    colors = ['#667eea' if row['Model_Type'] == 'Traditional' else '#764ba2' for _, row in top_10.iterrows()]
    
    bars = ax.barh(y_labels, top_10['Accuracy'].values*100, color=colors, alpha=0.8)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.2f}%', ha='left', va='center', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('Top 10 Model-Scaler Combinations', fontweight='bold', fontsize=14)
    ax.set_xlim([77, 79])
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

with tab5:
    st.subheader("📊 Comprehensive Statistics")
    
    # Overall stats
    col1, col2, col3, col4 = st.columns(4)
    
    all_scores = result_df.values.flatten()
    with col1:
        st.metric("Total Combinations", len(all_scores))
    with col2:
        st.metric("Average Accuracy", f"{np.mean(all_scores)*100:.2f}%")
    with col3:
        st.metric("Best Score", f"{np.max(all_scores)*100:.2f}%")
    with col4:
        st.metric("Worst Score", f"{np.min(all_scores)*100:.2f}%")
    
    st.divider()
    
    st.subheader("Statistics by Scaler")
    
    scaler_stats = []
    for scaler in result_df.columns:
        scores = result_df[scaler]
        scaler_stats.append({
            'Scaler': scaler,
            'Average': f"{scores.mean():.4f} ({scores.mean()*100:.2f}%)",
            'Maximum': f"{scores.max():.4f} ({scores.max()*100:.2f}%)",
            'Minimum': f"{scores.min():.4f} ({scores.min()*100:.2f}%)",
            'Std Dev': f"{scores.std():.4f}",
            'Best Model': scores.idxmax()
        })
    
    scaler_stats_df = pd.DataFrame(scaler_stats)
    st.dataframe(scaler_stats_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    st.subheader("Statistics by Model")
    
    col1, col2 = st.columns(2)
    
    for idx, model in enumerate(all_models):
        if idx % 2 == 0:
            col = col1
        else:
            col = col2
        
        with col:
            scores = result_df.loc[model]
            model_type = '🤖' if 'ANN' in model else '📊'
            st.write(f"### {model_type} {model}")
            st.write(f"**Average:** {scores.mean()*100:.2f}%")
            st.write(f"**Best:** {scores.idxmax()} ({scores.max()*100:.2f}%)")
            st.write(f"**Worst:** {scores.idxmin()} ({scores.min()*100:.2f}%)")
            st.write(f"**Range:** {(scores.max()-scores.min())*100:.2f}%")

with tab6:
    st.subheader("💡 Key Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🎯 Best Performing Models")
        st.markdown("""
        **Traditional Models:**
        1. **AdaBoost** - Consistent 77.92% across all scalers
        2. **Voting Classifier** - 77.14% average, improves with scaling
        3. **SVM** - 75.84% average, benefits from QuantileTransformer
        
        **Neural Networks:**
        1. **ANN-Large** - 74.16% average, best with MinMaxScaler (77.92%)
        2. **ANN-Deep** - 73.64% average, stable across scalers
        3. **ANN-Medium** - 70.65% average, best with PowerTransformer
        """)
    
    with col2:
        st.write("### 📈 Scaler Performance Rankings")
        st.markdown("""
        **Best Average Accuracy:**
        1. **PowerTransformer** - 75.32% (most consistent)
        2. **StandardScaler** - 74.86%
        3. **QuantileTransformer** - 73.79%
        4. **MinMaxScaler** - 72.87%
        5. **No Scaling** - 71.06%
        """)
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("### ✅ Best Combinations")
        st.markdown("""
        **Highest Accuracy:**
        - Neural Networks: ANN-Large + MinMaxScaler = 77.92%
        - Traditional: AdaBoost (any scaler) = 77.92%
        - Ensemble: Voting + PowerTransformer = 77.92%
        """)
    
    with col2:
        st.write("### 🔄 Scaler Impact")
        st.markdown("""
        **Most Affected Models:**
        - ANN-Small: -13.64% variance (needs scaling)
        - ANN-Medium: -12.99% variance
        - K-Nearest Neighbors: -8.44% variance
        
        **Least Affected Models:**
        - AdaBoost: 0% variance (scale-invariant)
        - Random Forest: -0.65% variance
        """)
    
    with col3:
        st.write("### 💡 Recommendations")
        st.markdown("""
        **For Production:**
        1. Use **AdaBoost** for reliability
        2. Use **ANN-Large** for neural networks
        3. Use **PowerTransformer** for scaling
        
        **For Best Accuracy:**
        - ANN-Large + MinMaxScaler = 77.92%
        - Voting Classifier + PowerTransformer = 77.92%
        """)

# Sidebar
st.sidebar.title("📊 Summary")

best_combo = configurations_df.iloc[0]
st.sidebar.metric("Best Configuration", f"{best_combo['Model']} + {best_combo['Scaler']}")
st.sidebar.metric("Best Accuracy", f"{best_combo['Accuracy']*100:.2f}%")

st.sidebar.divider()

st.sidebar.title("📈 Model Breakdown")

traditional_count = 9
ann_count = 5
st.sidebar.write(f"**Traditional Models:** {traditional_count}")
st.sidebar.write(f"**Neural Networks:** {ann_count}")
st.sidebar.write(f"**Total Scalers:** 5")
st.sidebar.write(f"**Total Combinations:** 70")

st.sidebar.divider()

st.sidebar.title("🔝 Best by Category")

trad_best = configurations_df[configurations_df['Model_Type'] == 'Traditional'].iloc[0]
ann_best = configurations_df[configurations_df['Model_Type'] == 'Neural Network'].iloc[0]

st.sidebar.write(f"**Traditional Best:**")
st.sidebar.write(f"{trad_best['Model']}")
st.sidebar.write(f"+ {trad_best['Scaler']}")
st.sidebar.write(f"{trad_best['Accuracy']*100:.2f}%")

st.sidebar.divider()

st.sidebar.write(f"**Neural Network Best:**")
st.sidebar.write(f"{ann_best['Model']}")
st.sidebar.write(f"+ {ann_best['Scaler']}")
st.sidebar.write(f"{ann_best['Accuracy']*100:.2f}%")

# Footer
st.divider()
st.markdown("""
**Comprehensive Analysis Summary:**
- **Models Evaluated:** 14 (9 Traditional + 5 Neural Network Architectures)
- **Scalers Tested:** 5 (No Scaling, StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer)
- **Total Combinations:** 70
- **Framework:** scikit-learn, Streamlit
- **Last Updated:** 2026-03-25
""")
