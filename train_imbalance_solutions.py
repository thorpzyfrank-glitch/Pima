import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('Pima  (1).csv')

# Display dataset info
print("=" * 80)
print("PIMA DIABETES CLASS IMBALANCE ANALYSIS")
print("=" * 80)

# Separate features and target
y = df['diabetes_class']
X = df.drop('diabetes_class', axis=1)

print(f"\nOriginal Dataset Shape: {X.shape}")
print(f"Original Class Distribution:")
print(y.value_counts())
print(f"Class Imbalance Ratio: {Counter(y)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining Set Original Distribution:")
print(f"  Minority Class: {sum(y_train == 1)}")
print(f"  Majority Class: {sum(y_train == 0)}")
print(f"  Imbalance Ratio: {sum(y_train == 0) / sum(y_train == 1):.2f}:1")

# Initialize results storage
results = []

# ============ NO RESAMPLING (BASELINE) ============
print("\n" + "=" * 80)
print("1. BASELINE - NO RESAMPLING")
print("=" * 80)

X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().fit_transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    results.append({
        'Technique': 'No Resampling',
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc,
        'Train Samples': len(X_train),
        'Minority (Train)': sum(y_train == 1),
        'Majority (Train)': sum(y_train == 0)
    })
    
    print(f"\n{model_name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")

# ============ SMOTE ============
print("\n" + "=" * 80)
print("2. SMOTE (Synthetic Minority Over-sampling Technique)")
print("=" * 80)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE, Training Set Distribution:")
print(f"  Minority Class: {sum(y_train_smote == 1)}")
print(f"  Majority Class: {sum(y_train_smote == 0)}")
print(f"  Total Samples: {len(X_train_smote)}")
print(f"  Balanced Ratio: 1:1")

X_train_smote_scaled = StandardScaler().fit_transform(X_train_smote)

for model_name, model in models.items():
    model.fit(X_train_smote_scaled, y_train_smote)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    results.append({
        'Technique': 'SMOTE',
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc,
        'Train Samples': len(X_train_smote),
        'Minority (Train)': sum(y_train_smote == 1),
        'Majority (Train)': sum(y_train_smote == 0)
    })
    
    print(f"\n{model_name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")

# ============ RANDOM OVER SAMPLER ============
print("\n" + "=" * 80)
print("3. RANDOM OVER SAMPLER")
print("=" * 80)

ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

print(f"\nAfter Random Over Sampler, Training Set Distribution:")
print(f"  Minority Class: {sum(y_train_ros == 1)}")
print(f"  Majority Class: {sum(y_train_ros == 0)}")
print(f"  Total Samples: {len(X_train_ros)}")
print(f"  Balanced Ratio: 1:1")

X_train_ros_scaled = StandardScaler().fit_transform(X_train_ros)

for model_name, model in models.items():
    model.fit(X_train_ros_scaled, y_train_ros)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    results.append({
        'Technique': 'Random Over Sampler',
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc,
        'Train Samples': len(X_train_ros),
        'Minority (Train)': sum(y_train_ros == 1),
        'Majority (Train)': sum(y_train_ros == 0)
    })
    
    print(f"\n{model_name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")

# ============ CREATE RESULTS DATAFRAME ============
results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("COMPREHENSIVE COMPARISON TABLE")
print("=" * 80)

# Create summary by technique
summary_by_technique = results_df.groupby('Technique')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].mean()
print("\n" + summary_by_technique.to_string())

# Save to CSV
results_df.to_csv('imbalance_solutions_comparison.csv', index=False)
print("\n✓ Results saved to 'imbalance_solutions_comparison.csv'")

# Create detailed report
with open('imbalance_solutions_results.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("PIMA DIABETES - CLASS IMBALANCE SOLUTIONS COMPARISON\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("ORIGINAL DATASET:\n")
    f.write(f"  Total Samples: {len(df)}\n")
    f.write(f"  Diabetes Cases (Positive): {sum(y == 1)}\n")
    f.write(f"  Non-Diabetes Cases (Negative): {sum(y == 0)}\n")
    f.write(f"  Imbalance Ratio: {sum(y == 0) / sum(y == 1):.2f}:1\n\n")
    
    f.write("RESAMPLING TECHNIQUES APPLIED:\n")
    f.write("1. No Resampling (Baseline)\n")
    f.write("2. SMOTE (Creates synthetic minority samples)\n")
    f.write("   - Training Samples: " + str(len(X_train_smote)) + "\n")
    f.write("3. Random Over Sampler (Duplicates minority samples)\n")
    f.write("   - Training Samples: " + str(len(X_train_ros)) + "\n\n")
    
    f.write("RESULTS BY TECHNIQUE:\n")
    f.write(summary_by_technique.to_string())
    f.write("\n\n")
    
    f.write("DETAILED RESULTS:\n")
    f.write(results_df.to_string(index=False))

print("✓ Detailed report saved to 'imbalance_solutions_results.txt'")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
