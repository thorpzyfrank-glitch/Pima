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

print("="*100)
print("CLASSIFICATION MODELS WITH NEURAL NETWORKS - COMPREHENSIVE ANALYSIS")
print("="*100)
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

# Define neural network architectures
ann_architectures = {
    'ANN-Small': (50,),
    'ANN-Medium': (100, 50),
    'ANN-Large': (256, 128, 64),
    'ANN-Deep': (200, 150, 100, 50),
    'ANN-Wide': (500,),
}

# Define all models including multiple ANN architectures
def get_models():
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15),
        'SVM': SVC(kernel='rbf', random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, max_depth=5),
        'AdaBoost': AdaBoostClassifier(random_state=42, n_estimators=50),
        'Voting Classifier': VotingClassifier(
            estimators=[('lr', LogisticRegression(max_iter=1000, random_state=42)),
                       ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                       ('svm', SVC(kernel='rbf', random_state=42, probability=True))],
            voting='soft'
        )
    }
    
    # Add neural network architectures
    for ann_name, hidden_layers in ann_architectures.items():
        models[ann_name] = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=1000, random_state=42, early_stopping=True)
    
    return models

# Store all results
all_results = {}
all_models_list = list(get_models().keys())

print(f"Total Models: {len(all_models_list)}")
print(f"Models: {', '.join(all_models_list)}\n")

# Train models with each scaler
for scaler_name, scaler in scalers.items():
    print(f"\n{'='*100}")
    print(f"Training with: {scaler_name}")
    print(f"{'='*100}\n")
    
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
        print(f"{model_name:.<35} {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    all_results[scaler_name] = scaler_results
    print()

# Save results
with open('comprehensive_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

print("\n" + "="*100)
print("COMPREHENSIVE RESULTS TABLE")
print("="*100 + "\n")

# Create comprehensive comparison table
result_df = pd.DataFrame(all_results)
result_df = result_df.reindex(all_models_list)

print(result_df.to_string())

# Save as CSV
result_df.to_csv('comprehensive_comparison_table.csv')
print("\n\n✓ Results saved to 'comprehensive_comparison_table.csv'")

# Save as formatted CSV with percentages
result_df_pct = result_df.copy()
for col in result_df_pct.columns:
    result_df_pct[col] = result_df_pct[col].apply(lambda x: f"{x*100:.2f}%")

result_df_pct.to_csv('comprehensive_comparison_table_pct.csv')
print("✓ Formatted results saved to 'comprehensive_comparison_table_pct.csv'")

# Calculate statistics
print("\n" + "="*100)
print("STATISTICS BY SCALER")
print("="*100 + "\n")

for scaler_name in result_df.columns:
    scores = result_df[scaler_name]
    print(f"\n{scaler_name}:")
    print(f"  Average Accuracy: {scores.mean():.4f} ({scores.mean()*100:.2f}%)")
    print(f"  Maximum Accuracy: {scores.max():.4f} ({scores.max()*100:.2f}%)")
    print(f"  Minimum Accuracy: {scores.min():.4f} ({scores.min()*100:.2f}%)")
    print(f"  Std Deviation: {scores.std():.4f}")
    print(f"  Best Model: {scores.idxmax()}")

# Calculate statistics by model category
print("\n" + "="*100)
print("TRADITIONAL MODELS PERFORMANCE")
print("="*100 + "\n")

traditional_models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM',
                     'K-Nearest Neighbors', 'Naive Bayes', 'Gradient Boosting',
                     'AdaBoost', 'Voting Classifier']

traditional_df = result_df.loc[traditional_models]
for model_name in traditional_models:
    scores = result_df.loc[model_name]
    print(f"\n{model_name}:")
    print(f"  Average Accuracy: {scores.mean():.4f} ({scores.mean()*100:.2f}%)")
    print(f"  Best Scaler: {scores.idxmax()} ({scores.max():.4f})")
    print(f"  Worst Scaler: {scores.idxmin()} ({scores.min():.4f})")
    print(f"  Range: {(scores.max() - scores.min())*100:.2f}%")

# Analyze ANN architectures
print("\n" + "="*100)
print("ARTIFICIAL NEURAL NETWORK (ANN) ARCHITECTURES PERFORMANCE")
print("="*100 + "\n")

for ann_name in ann_architectures.keys():
    scores = result_df.loc[ann_name]
    print(f"\n{ann_name}:")
    print(f"  Average Accuracy: {scores.mean():.4f} ({scores.mean()*100:.2f}%)")
    print(f"  Best Scaler: {scores.idxmax()} ({scores.max():.4f})")
    print(f"  Worst Scaler: {scores.idxmin()} ({scores.min():.4f})")
    print(f"  Range: {(scores.max() - scores.min())*100:.2f}%")

# Find best configurations
print("\n" + "="*100)
print("BEST MODEL-SCALER CONFIGURATIONS")
print("="*100 + "\n")

# Flatten and find best
all_combinations = []
for scaler_name, model_scores in all_results.items():
    for model_name, accuracy in model_scores.items():
        all_combinations.append({
            'Model': model_name,
            'Scaler': scaler_name,
            'Accuracy': accuracy,
            'Model_Type': 'Neural Network' if 'ANN' in model_name else 'Traditional'
        })

combinations_df = pd.DataFrame(all_combinations)
combinations_df = combinations_df.sort_values('Accuracy', ascending=False)

with open('best_configurations.pkl', 'wb') as f:
    pickle.dump(combinations_df, f)

print("\nTop 15 Best Overall Combinations:")
for idx, row in combinations_df.head(15).iterrows():
    print(f"{combinations_df.index.get_loc(idx)+1:2}. {row['Model']:.<25} + {row['Scaler']:<20} = {row['Accuracy']:.4f} ({row['Accuracy']*100:.2f}%)")

print("\n\nTop 5 Best Neural Network Configurations:")
ann_combos = combinations_df[combinations_df['Model_Type'] == 'Neural Network'].head(5)
for idx, (i, row) in enumerate(ann_combos.iterrows(), 1):
    print(f"{idx}. {row['Model']:.<25} + {row['Scaler']:<20} = {row['Accuracy']:.4f} ({row['Accuracy']*100:.2f}%)")

print("\n\nTop 5 Best Traditional Model Configurations:")
trad_combos = combinations_df[combinations_df['Model_Type'] == 'Traditional'].head(5)
for idx, (i, row) in enumerate(trad_combos.iterrows(), 1):
    print(f"{idx}. {row['Model']:.<25} + {row['Scaler']:<20} = {row['Accuracy']:.4f} ({row['Accuracy']*100:.2f}%)")

print("\n" + "="*100)
print("SUMMARY")
print("="*100)
print(f"\nTotal Models Evaluated: {len(all_models_list)}")
print(f"Total Scalers: {len(scalers)}")
print(f"Total Combinations: {len(all_models_list) * len(scalers)}")
print(f"\nBest Overall Accuracy: {combinations_df.iloc[0]['Accuracy']:.4f} ({combinations_df.iloc[0]['Accuracy']*100:.2f}%)")
print(f"Best Configuration: {combinations_df.iloc[0]['Model']} + {combinations_df.iloc[0]['Scaler']}")
print(f"\nAverage Accuracy Across All Combinations: {combinations_df['Accuracy'].mean():.4f} ({combinations_df['Accuracy'].mean()*100:.2f}%)")
print(f"Standard Deviation: {combinations_df['Accuracy'].std():.4f}")
