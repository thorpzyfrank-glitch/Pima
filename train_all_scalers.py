import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Import all models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

# Load and prepare data
df = pd.read_csv('Pima.csv')
X = df.drop('diabetes_class', axis=1)
y = df['diabetes_class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("="*80)
print("CLASSIFICATION MODELS WITH DIFFERENT SCALERS")
print("="*80)
print(f"\nData Shape: {X.shape}")
print(f"Train Set: {X_train.shape[0]}, Test Set: {X_test.shape[0]}\n")

# Define scalers
scalers = {
    'No Scaling': None,
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'PowerTransformer': PowerTransformer(method='yeo-johnson'),
    'QuantileTransformer': QuantileTransformer(output_distribution='uniform', random_state=42)
}

# Define models
def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15),
        'SVM': SVC(kernel='rbf', random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, max_depth=5),
        'AdaBoost': AdaBoostClassifier(random_state=42, n_estimators=50),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
        'Voting Classifier': VotingClassifier(
            estimators=[('lr', LogisticRegression(max_iter=1000, random_state=42)),
                       ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                       ('svm', SVC(kernel='rbf', random_state=42, probability=True))],
            voting='soft'
        )
    }

# Store all results
all_results = {}

# Train models with each scaler
for scaler_name, scaler in scalers.items():
    print(f"\n{'='*80}")
    print(f"Training with: {scaler_name}")
    print(f"{'='*80}\n")
    
    # Apply scaler
    if scaler is None:
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
    else:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    scaler_results = {}
    models = get_models()
    
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        scaler_results[model_name] = accuracy
        print(f"{model_name:.<30} {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    all_results[scaler_name] = scaler_results

# Save results
with open('scaler_comparison_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

print("\n" + "="*80)
print("SUMMARY: ACCURACY SCORES BY SCALER AND MODEL")
print("="*80 + "\n")

# Create comprehensive comparison table
result_df = pd.DataFrame(all_results)
result_df = result_df.reindex(['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM',
                               'K-Nearest Neighbors', 'Naive Bayes', 'Gradient Boosting',
                               'AdaBoost', 'Neural Network', 'Voting Classifier'])

print(result_df.to_string())

# Save as CSV
result_df.to_csv('scaler_comparison_table.csv')
print("\n✓ Results saved to 'scaler_comparison_table.csv'")

# Calculate statistics
print("\n" + "="*80)
print("STATISTICS BY SCALER")
print("="*80 + "\n")

for scaler_name in result_df.columns:
    scores = result_df[scaler_name]
    print(f"\n{scaler_name}:")
    print(f"  Average Accuracy: {scores.mean():.4f} ({scores.mean()*100:.2f}%)")
    print(f"  Maximum Accuracy: {scores.max():.4f} ({scores.max()*100:.2f}%)")
    print(f"  Minimum Accuracy: {scores.min():.4f} ({scores.min()*100:.2f}%)")
    print(f"  Std Deviation: {scores.std():.4f}")
    print(f"  Best Model: {scores.idxmax()}")

# Calculate statistics by model
print("\n" + "="*80)
print("STATISTICS BY MODEL")
print("="*80 + "\n")

for model_name in result_df.index:
    scores = result_df.loc[model_name]
    print(f"\n{model_name}:")
    print(f"  Average Accuracy: {scores.mean():.4f} ({scores.mean()*100:.2f}%)")
    print(f"  Maximum Accuracy: {scores.max():.4f} ({scores.max()*100:.2f}%) - {scores.idxmax()}")
    print(f"  Minimum Accuracy: {scores.min():.4f} ({scores.min()*100:.2f}%) - {scores.idxmin()}")
    print(f"  Range: {(scores.max() - scores.min())*100:.2f}%")

# Find best combination
print("\n" + "="*80)
print("BEST COMBINATIONS")
print("="*80 + "\n")

# Flatten and find best
all_combinations = []
for scaler_name, model_scores in all_results.items():
    for model_name, accuracy in model_scores.items():
        all_combinations.append({
            'Scaler': scaler_name,
            'Model': model_name,
            'Accuracy': accuracy
        })

combinations_df = pd.DataFrame(all_combinations)
combinations_df = combinations_df.sort_values('Accuracy', ascending=False)

print("\nTop 10 Best Combinations:")
for idx, row in combinations_df.head(10).iterrows():
    print(f"{combinations_df.index.get_loc(idx)+1:2}. {row['Model']:25} + {row['Scaler']:20} = {row['Accuracy']:.4f} ({row['Accuracy']*100:.2f}%)")

# Save combinations
with open('best_combinations.pkl', 'wb') as f:
    pickle.dump(combinations_df, f)
