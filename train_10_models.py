import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# Load dataset
print("Loading Pima Indians Diabetes Dataset...")
df = pd.read_csv('Pima  (1).csv')
print(f"Dataset shape: {df.shape}")

# Separate features and target
X = df.drop('diabetes_class', axis=1)
y = df['diabetes_class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Features: {X.shape[1]}")
print("\n" + "="*80)
print("Training 10 Different Classification Algorithms...")
print("="*80 + "\n")

# Dictionary to store models and results
models = {}
results = []

# 1. Logistic Regression
print("1. Training Logistic Regression...")
models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=42)
models['Logistic Regression'].fit(X_train_scaled, y_train)
y_pred = models['Logistic Regression'].predict(X_test_scaled)
results.append({
    'Model': 'Logistic Regression',
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred)
})

# 2. Random Forest
print("2. Training Random Forest...")
models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
models['Random Forest'].fit(X_train_scaled, y_train)
y_pred = models['Random Forest'].predict(X_test_scaled)
results.append({
    'Model': 'Random Forest',
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred)
})

# 3. Gradient Boosting
print("3. Training Gradient Boosting...")
models['Gradient Boosting'] = GradientBoostingClassifier(n_estimators=100, random_state=42)
models['Gradient Boosting'].fit(X_train_scaled, y_train)
y_pred = models['Gradient Boosting'].predict(X_test_scaled)
results.append({
    'Model': 'Gradient Boosting',
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred)
})

# 4. Support Vector Machine (SVM)
print("4. Training Support Vector Machine...")
models['SVM'] = SVC(kernel='rbf', probability=True, random_state=42)
models['SVM'].fit(X_train_scaled, y_train)
y_pred = models['SVM'].predict(X_test_scaled)
results.append({
    'Model': 'Support Vector Machine',
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred)
})

# 5. K-Nearest Neighbors
print("5. Training K-Nearest Neighbors...")
models['KNN'] = KNeighborsClassifier(n_neighbors=5)
models['KNN'].fit(X_train_scaled, y_train)
y_pred = models['KNN'].predict(X_test_scaled)
results.append({
    'Model': 'K-Nearest Neighbors',
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred)
})

# 6. Naive Bayes
print("6. Training Naive Bayes...")
models['Naive Bayes'] = GaussianNB()
models['Naive Bayes'].fit(X_train_scaled, y_train)
y_pred = models['Naive Bayes'].predict(X_test_scaled)
results.append({
    'Model': 'Naive Bayes',
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred)
})

# 7. Decision Tree
print("7. Training Decision Tree...")
models['Decision Tree'] = DecisionTreeClassifier(random_state=42)
models['Decision Tree'].fit(X_train_scaled, y_train)
y_pred = models['Decision Tree'].predict(X_test_scaled)
results.append({
    'Model': 'Decision Tree',
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred)
})

# 8. AdaBoost
print("8. Training AdaBoost...")
models['AdaBoost'] = AdaBoostClassifier(n_estimators=100, random_state=42)
models['AdaBoost'].fit(X_train_scaled, y_train)
y_pred = models['AdaBoost'].predict(X_test_scaled)
results.append({
    'Model': 'AdaBoost',
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred)
})

# 9. XGBoost
print("9. Training XGBoost...")
models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
models['XGBoost'].fit(X_train_scaled, y_train)
y_pred = models['XGBoost'].predict(X_test_scaled)
results.append({
    'Model': 'XGBoost',
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred)
})

# 10. Neural Network (MLP)
print("10. Training Neural Network (MLP)...")
models['MLP'] = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
models['MLP'].fit(X_train_scaled, y_train)
y_pred = models['MLP'].predict(X_test_scaled)
results.append({
    'Model': 'Neural Network (MLP)',
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred)
})

# Create comparison table
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)

print("\n" + "="*100)
print("MODEL COMPARISON TABLE - CLASSIFICATION ALGORITHMS")
print("="*100 + "\n")

# Display table with formatting
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print(results_df.to_string(index=False))

# Summary statistics
print("\n" + "="*100)
print("SUMMARY STATISTICS")
print("="*100)
print(f"Best Model: {results_df.iloc[0]['Model']} with {results_df.iloc[0]['Accuracy']:.4f} accuracy")
print(f"Worst Model: {results_df.iloc[-1]['Model']} with {results_df.iloc[-1]['Accuracy']:.4f} accuracy")
print(f"Average Accuracy: {results_df['Accuracy'].mean():.4f}")
print(f"Accuracy Range: {results_df['Accuracy'].min():.4f} - {results_df['Accuracy'].max():.4f}")

# Save to CSV
results_df.to_csv('model_comparison_table.csv', index=False)
print("\n✅ Model comparison table saved to 'model_comparison_table.csv'")

# Create a detailed report
print("\n" + "="*100)
print("DETAILED RESULTS (Best to Worst)")
print("="*100 + "\n")
for idx, row in results_df.iterrows():
    print(f"{idx+1}. {row['Model']}")
    print(f"   Accuracy:  {row['Accuracy']:.4f} ({row['Accuracy']*100:.2f}%)")
    print(f"   Precision: {row['Precision']:.4f}")
    print(f"   Recall:    {row['Recall']:.4f}")
    print(f"   F1-Score:  {row['F1-Score']:.4f}")
    print()
