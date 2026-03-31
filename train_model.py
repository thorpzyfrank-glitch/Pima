import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# Load data
df = pd.read_csv('Pima  (1).csv')

# Features and target
X = df.drop('diabetes_class', axis=1)
y = df['diabetes_class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Gradient Boosting model (high accuracy)
print("Training Gradient Boosting Classifier...")
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    subsample=0.8
)
gb_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = gb_model.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n=== Model Performance ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Save model and scaler
joblib.dump(gb_model, 'pima_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns, 'feature_names.pkl')

print("\nModel and scaler saved successfully!")
