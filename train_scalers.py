import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("Loading dataset...")
df = pd.read_csv('Pima  (1).csv')

# Prepare data
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing techniques
scalers = {
    'MinMax Scaler': MinMaxScaler(),
    'Standard Scaler': StandardScaler(),
    'Robust Scaler': RobustScaler(),
    'MaxAbs Scaler': MaxAbsScaler(),
    'Log Transformation': None  # Special handling for log transformation
}

results = []

print("\n" + "="*80)
print("Training AdaBoost with 5 Different Preprocessing Techniques")
print("="*80 + "\n")

# Train model with each preprocessing technique
for scaler_name, scaler in scalers.items():
    print(f"Processing: {scaler_name}...")
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if scaler_name == 'Log Transformation':
        # Apply log transformation (add 1 to avoid log(0))
        X_train_scaled = np.log1p(X_train_scaled)
        X_test_scaled = np.log1p(X_test_scaled)
    else:
        # Apply scaler
        X_train_scaled = scaler.fit_transform(X_train_scaled)
        X_test_scaled = scaler.transform(X_test_scaled)
    
    # Train AdaBoost model
    model = AdaBoostClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    results.append({
        'Preprocessing Technique': scaler_name,
        'Accuracy': f"{accuracy*100:.2f}%",
        'Precision': f"{precision*100:.2f}%",
        'Recall': f"{recall*100:.2f}%",
        'F1-Score': f"{f1*100:.2f}%",
        'Accuracy_Score': accuracy
    })
    
    print(f"  ✓ Accuracy: {accuracy*100:.2f}%")

# Create results dataframe
results_df = pd.DataFrame(results)

# Sort by accuracy (descending)
results_df_sorted = results_df.sort_values('Accuracy_Score', ascending=False).reset_index(drop=True)

# Remove the sorting column for display
display_df = results_df_sorted[['Preprocessing Technique', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]
display_df.index = display_df.index + 1

# Save to CSV
results_df_sorted[['Preprocessing Technique', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].to_csv('scaler_comparison_table.csv', index=False)

# Print results
print("\n" + "="*80)
print("PREPROCESSING TECHNIQUES COMPARISON RESULTS")
print("="*80)
print(display_df.to_string())
print("="*80)

# Save detailed results
with open('scaler_comparison_results.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("PIMA DIABETES: 5 PREPROCESSING TECHNIQUES COMPARISON\n")
    f.write("="*80 + "\n\n")
    
    f.write("DATASET INFORMATION:\n")
    f.write(f"  - Training Samples: {len(X_train)}\n")
    f.write(f"  - Testing Samples: {len(X_test)}\n")
    f.write(f"  - Total Samples: {len(X)}\n")
    f.write(f"  - Features: {X.shape[1]}\n\n")
    
    f.write("MODEL CONFIGURATION:\n")
    f.write("  - Algorithm: AdaBoost Classifier\n")
    f.write("  - Estimators: 200\n")
    f.write("  - Random State: 42\n\n")
    
    f.write("="*80 + "\n")
    f.write("PREPROCESSING TECHNIQUES COMPARISON TABLE\n")
    f.write("="*80 + "\n\n")
    f.write(display_df.to_string())
    f.write("\n\n")
    
    f.write("="*80 + "\n")
    f.write("DETAILED RESULTS\n")
    f.write("="*80 + "\n\n")
    
    for idx, row in results_df_sorted.iterrows():
        f.write(f"{idx+1}. {row['Preprocessing Technique']}\n")
        f.write(f"   - Accuracy:  {row['Accuracy']}\n")
        f.write(f"   - Precision: {row['Precision']}\n")
        f.write(f"   - Recall:    {row['Recall']}\n")
        f.write(f"   - F1-Score:  {row['F1-Score']}\n\n")
    
    f.write("="*80 + "\n")
    f.write("KEY INSIGHTS\n")
    f.write("="*80 + "\n\n")
    
    best_scaler = results_df_sorted.iloc[0]
    worst_scaler = results_df_sorted.iloc[-1]
    
    f.write(f"✓ BEST PREPROCESSING TECHNIQUE:\n")
    f.write(f"  {best_scaler['Preprocessing Technique']} - {best_scaler['Accuracy']}\n\n")
    
    f.write(f"✗ WORST PREPROCESSING TECHNIQUE:\n")
    f.write(f"  {worst_scaler['Preprocessing Technique']} - {worst_scaler['Accuracy']}\n\n")
    
    avg_accuracy = np.mean([float(row['Accuracy'].rstrip('%')) for row in results])
    f.write(f"📊 AVERAGE ACCURACY: {avg_accuracy:.2f}%\n")

print("\n✓ Results saved to:")
print("  - scaler_comparison_table.csv")
print("  - scaler_comparison_results.txt")
