import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Classification Models - Scaled vs Unscaled",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 30px;
    }
    .best-model-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 20px 0;
    }
    .model-name {
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .accuracy-score {
        font-size: 48px;
        font-weight: bold;
        margin: 20px 0;
    }
    .improvement-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 15px;
        border-radius: 5px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .decrease-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 15px;
        border-radius: 5px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">🤖 Classification Models Comparison</div>', unsafe_allow_html=True)
st.markdown("**Pima Indians Diabetes - With & Without MinMaxScaler**")

# Load accuracy data
with open('accuracies.pkl', 'rb') as f:
    accuracies_unscaled = pickle.load(f)

with open('accuracies_scaled.pkl', 'rb') as f:
    accuracies_scaled = pickle.load(f)

# Create comparison dataframe
comparison_data = []
for model_name in accuracies_unscaled.keys():
    unscaled = accuracies_unscaled[model_name]
    scaled = accuracies_scaled[model_name]
    improvement = (scaled - unscaled) * 100
    comparison_data.append({
        "Model": model_name,
        "Unscaled": unscaled,
        "Scaled": scaled,
        "Improvement (%)": improvement
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Scaled', ascending=False)

# Display best models
col1, col2 = st.columns(2)

with col1:
    best_unscaled_model = max(accuracies_unscaled, key=accuracies_unscaled.get)
    best_unscaled_acc = accuracies_unscaled[best_unscaled_model]
    st.markdown(f"""
        <div class="best-model-box">
            <div class="model-name">📊 Without Scaling</div>
            <div>{best_unscaled_model}</div>
            <div class="accuracy-score">{best_unscaled_acc*100:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    best_scaled_model = max(accuracies_scaled, key=accuracies_scaled.get)
    best_scaled_acc = accuracies_scaled[best_scaled_model]
    st.markdown(f"""
        <div class="best-model-box">
            <div class="model-name">📊 With MinMaxScaler</div>
            <div>{best_scaled_model}</div>
            <div class="accuracy-score">{best_scaled_acc*100:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Comparison", "📈 Scaled Results", "🔄 Impact Analysis", "📋 Detailed Metrics"])

with tab1:
    st.subheader("Side-by-Side Model Comparison")
    
    # Display comparison table
    display_df = comparison_df.copy()
    display_df['Unscaled'] = display_df['Unscaled'].apply(lambda x: f"{x:.4f} ({x*100:.2f}%)")
    display_df['Scaled'] = display_df['Scaled'].apply(lambda x: f"{x:.4f} ({x*100:.2f}%)")
    display_df['Improvement (%)'] = display_df['Improvement (%)'].apply(lambda x: f"{x:+.2f}%")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Show improvements/decreases
    st.subheader("Impact of MinMaxScaler on Each Model")
    for idx, row in comparison_df.iterrows():
        improvement = row['Improvement (%)']
        if improvement >= 0:
            st.markdown(f"""
                <div class="improvement-box">
                    <strong>{row['Model']}</strong><br>
                    +{improvement:.2f}% improvement ({row['Unscaled']*100:.2f}% → {row['Scaled']*100:.2f}%)
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="decrease-box">
                    <strong>{row['Model']}</strong><br>
                    {improvement:.2f}% decrease ({row['Unscaled']*100:.2f}% → {row['Scaled']*100:.2f}%)
                </div>
            """, unsafe_allow_html=True)

with tab2:
    st.subheader("Model Rankings - With MinMaxScaler")
    
    # Create ranking with scaled data
    sorted_scaled = sorted(accuracies_scaled.items(), key=lambda x: x[1], reverse=True)
    ranking_data = []
    for rank, (model, acc) in enumerate(sorted_scaled, 1):
        ranking_data.append({
            "Rank": rank,
            "Model Name": model,
            "Accuracy": f"{acc:.4f}",
            "Percentage": f"{acc*100:.2f}%"
        })
    
    ranking_df = pd.DataFrame(ranking_data)
    st.dataframe(ranking_df, use_container_width=True, hide_index=True)
    
    st.subheader("Accuracy Progress Bars (Scaled)")
    for i, (model_name, accuracy) in enumerate(sorted_scaled, 1):
        col1, col2, col3 = st.columns([2, 3, 1])
        with col1:
            st.write(f"**{i}. {model_name}**")
        with col2:
            st.progress(accuracy)
        with col3:
            st.write(f"`{accuracy*100:.2f}%`")

with tab3:
    st.subheader("Impact Analysis - MinMaxScaler Effect")
    
    # Create comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data
    models_list = list(accuracies_unscaled.keys())
    unscaled_acc = [accuracies_unscaled[m]*100 for m in models_list]
    scaled_acc = [accuracies_scaled[m]*100 for m in models_list]
    
    x = np.arange(len(models_list))
    width = 0.35
    
    # Chart 1: Side-by-side comparison
    bars1 = ax1.bar(x - width/2, unscaled_acc, width, label='Without Scaling', color='#667eea', alpha=0.8)
    bars2 = ax1.bar(x + width/2, scaled_acc, width, label='With MinMaxScaler', color='#764ba2', alpha=0.8)
    
    ax1.set_xlabel('Models', fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Model Accuracy: Unscaled vs Scaled', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_list, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim([60, 85])
    ax1.grid(axis='y', alpha=0.3)
    
    # Chart 2: Improvement percentage
    improvements = [(accuracies_scaled[m] - accuracies_unscaled[m])*100 for m in models_list]
    colors = ['#84fab0' if imp >= 0 else '#fa709a' for imp in improvements]
    
    ax2.barh(models_list, improvements, color=colors, alpha=0.8)
    ax2.set_xlabel('Accuracy Change (%)', fontweight='bold')
    ax2.set_title('Impact of MinMaxScaler on Accuracy', fontweight='bold', fontsize=12)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(improvements):
        ax2.text(v + 0.1 if v >= 0 else v - 0.1, i, f'{v:+.2f}%', 
                va='center', ha='left' if v >= 0 else 'right', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show statistics
    st.subheader("Scaling Impact Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    improved_count = sum(1 for m in models_list if accuracies_scaled[m] > accuracies_unscaled[m])
    decreased_count = sum(1 for m in models_list if accuracies_scaled[m] < accuracies_unscaled[m])
    unchanged_count = sum(1 for m in models_list if accuracies_scaled[m] == accuracies_unscaled[m])
    avg_improvement = np.mean(improvements)
    max_improvement = max(improvements)
    max_decrease = min(improvements)
    
    with col1:
        st.metric("Models Improved", improved_count)
    with col2:
        st.metric("Models Decreased", decreased_count)
    with col3:
        st.metric("Models Unchanged", unchanged_count)
    with col4:
        st.metric("Avg Change", f"{avg_improvement:+.2f}%")
    
    st.write(f"**Maximum Improvement:** {max_improvement:+.2f}%")
    st.write(f"**Maximum Decrease:** {max_decrease:+.2f}%")

with tab4:
    st.subheader("📋 Detailed Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Unscaled Data**")
        avg_unscaled = np.mean(list(accuracies_unscaled.values()))
        st.metric("Average Accuracy", f"{avg_unscaled:.4f} ({avg_unscaled*100:.2f}%)")
    
    with col2:
        st.write("**Scaled Data (MinMaxScaler)**")
        avg_scaled = np.mean(list(accuracies_scaled.values()))
        st.metric("Average Accuracy", f"{avg_scaled:.4f} ({avg_scaled*100:.2f}%)")
    
    with col3:
        st.write("**Overall Impact**")
        overall_change = (avg_scaled - avg_unscaled) * 100
        st.metric("Average Change", f"{overall_change:+.2f}%")
    
    st.divider()
    
    # Detailed breakdown for each model
    st.subheader("Individual Model Metrics")
    
    for idx, row in comparison_df.iterrows():
        with st.expander(f"{row['Model']} - {row['Scaled']*100:.2f}% (Scaled)", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write(f"**Unscaled Accuracy**\n{row['Unscaled']:.4f} ({row['Unscaled']*100:.2f}%)")
            with col2:
                st.write(f"**Scaled Accuracy**\n{row['Scaled']:.4f} ({row['Scaled']*100:.2f}%)")
            with col3:
                st.write(f"**Absolute Change**\n{(row['Scaled']-row['Unscaled'])*100:+.2f}%")
            with col4:
                st.write(f"**Relative Impact**\n{row['Improvement (%)']:+.2f}%")

# Sidebar information
st.sidebar.title("📋 Configuration")
st.sidebar.write("**Dataset:** Pima Indians Diabetes")
st.sidebar.write("**Total Samples:** 768")
st.sidebar.write("**Features:** 8")
st.sidebar.write("**Train-Test Split:** 80-20")
st.sidebar.divider()

st.sidebar.title("🔧 Scaling Method")
st.sidebar.write("**Scaler:** MinMaxScaler")
st.sidebar.write("**Feature Range:** [0, 1]")
st.sidebar.write("**Formula:** (X - X_min) / (X_max - X_min)")
st.sidebar.divider()

st.sidebar.title("🏆 Best Models")
st.sidebar.subheader("Unscaled")
st.sidebar.write(f"{best_unscaled_model}: {best_unscaled_acc*100:.2f}%")

st.sidebar.subheader("Scaled")
st.sidebar.write(f"{best_scaled_model}: {best_scaled_acc*100:.2f}%")

# Footer
st.divider()
st.markdown("""
**Status:** ✅ All 10 Models Trained with MinMaxScaler  
**Last Updated:** 2026-03-25  
**Framework:** scikit-learn, Streamlit  
**Scaling Impact:** Positive for some models, demonstrating the importance of feature preprocessing
""")
