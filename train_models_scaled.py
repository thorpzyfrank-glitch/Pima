import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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

# Apply MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("="*60)
print("CLASSIFICATION MODELS WITH MinMaxScaler")
print("="*60)
print(f"\nData Shape: {X_scaled.shape}")
print(f"Train Set: {X_train.shape[0]}, Test Set: {X_test.shape[0]}")
print(f"\nFeature Scaling: MinMaxScaler (0-1 range)")
print("\n" + "="*60)

# Dictionary to store all models
models_scaled = {}
accuracies_scaled = {}

# 1. Logistic Regression
print("\n[1/10] Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
models_scaled['Logistic Regression'] = lr
accuracies_scaled['Logistic Regression'] = lr_accuracy
print(f"✓ Logistic Regression: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")

# 2. Decision Tree Classifier
print("[2/10] Training Decision Tree...")
dt = DecisionTreeClassifier(random_state=42, max_depth=10)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
models_scaled['Decision Tree'] = dt
accuracies_scaled['Decision Tree'] = dt_accuracy
print(f"✓ Decision Tree: {dt_accuracy:.4f} ({dt_accuracy*100:.2f}%)")

# 3. Random Forest Classifier
print("[3/10] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
models_scaled['Random Forest'] = rf
accuracies_scaled['Random Forest'] = rf_accuracy
print(f"✓ Random Forest: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")

# 4. Support Vector Machine (SVM)
print("[4/10] Training Support Vector Machine...")
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
models_scaled['SVM'] = svm
accuracies_scaled['SVM'] = svm_accuracy
print(f"✓ Support Vector Machine: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")

# 5. K-Nearest Neighbors
print("[5/10] Training K-Nearest Neighbors...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
models_scaled['K-Nearest Neighbors'] = knn
accuracies_scaled['K-Nearest Neighbors'] = knn_accuracy
print(f"✓ K-Nearest Neighbors: {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)")

# 6. Naive Bayes
print("[6/10] Training Naive Bayes...")
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)
models_scaled['Naive Bayes'] = nb
accuracies_scaled['Naive Bayes'] = nb_accuracy
print(f"✓ Naive Bayes: {nb_accuracy:.4f} ({nb_accuracy*100:.2f}%)")

# 7. Gradient Boosting Classifier
print("[7/10] Training Gradient Boosting...")
gb = GradientBoostingClassifier(random_state=42, max_depth=5)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
models_scaled['Gradient Boosting'] = gb
accuracies_scaled['Gradient Boosting'] = gb_accuracy
print(f"✓ Gradient Boosting: {gb_accuracy:.4f} ({gb_accuracy*100:.2f}%)")

# 8. AdaBoost Classifier
print("[8/10] Training AdaBoost...")
ab = AdaBoostClassifier(random_state=42, n_estimators=50)
ab.fit(X_train, y_train)
ab_pred = ab.predict(X_test)
ab_accuracy = accuracy_score(y_test, ab_pred)
models_scaled['AdaBoost'] = ab
accuracies_scaled['AdaBoost'] = ab_accuracy
print(f"✓ AdaBoost: {ab_accuracy:.4f} ({ab_accuracy*100:.2f}%)")

# 9. Neural Network (MLPClassifier)
print("[9/10] Training Neural Network (MLP)...")
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_pred)
models_scaled['Neural Network'] = mlp
accuracies_scaled['Neural Network'] = mlp_accuracy
print(f"✓ Neural Network (MLP): {mlp_accuracy:.4f} ({mlp_accuracy*100:.2f}%)")

# 10. Voting Classifier (Ensemble)
print("[10/10] Training Voting Classifier...")
voting = VotingClassifier(
    estimators=[('lr', LogisticRegression(max_iter=1000, random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('svm', SVC(kernel='rbf', random_state=42, probability=True))],
    voting='soft'
)
voting.fit(X_train, y_train)
voting_pred = voting.predict(X_test)
voting_accuracy = accuracy_score(y_test, voting_pred)
models_scaled['Voting Classifier'] = voting
accuracies_scaled['Voting Classifier'] = voting_accuracy
print(f"✓ Voting Classifier: {voting_accuracy:.4f} ({voting_accuracy*100:.2f}%)")

# Save accuracy scores
with open('accuracies_scaled.pkl', 'wb') as f:
    pickle.dump(accuracies_scaled, f)

# Save scaler
with open('minmax_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n" + "="*60)
print("All models trained with MinMaxScaler successfully!")
print("="*60)

# Find best model
best_model = max(accuracies_scaled, key=accuracies_scaled.get)
best_accuracy = accuracies_scaled[best_model]
print(f"\n🏆 Best Model: {best_model}")
print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# Sort and display all models by accuracy
print("\n" + "="*60)
print("Model Rankings by Accuracy (With MinMaxScaler):")
print("="*60)
sorted_models = sorted(accuracies_scaled.items(), key=lambda x: x[1], reverse=True)
for i, (model_name, accuracy) in enumerate(sorted_models, 1):
    print(f"{i}. {model_name:.<30} {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n" + "="*60)
print("Summary Statistics:")
print("="*60)
avg_accuracy = sum([acc for _, acc in sorted_models]) / len(sorted_models)
max_accuracy = max([acc for _, acc in sorted_models])
min_accuracy = min([acc for _, acc in sorted_models])
print(f"Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
print(f"Maximum Accuracy: {max_accuracy:.4f} ({max_accuracy*100:.2f}%)")
print(f"Minimum Accuracy: {min_accuracy:.4f} ({min_accuracy*100:.2f}%)")
print(f"Accuracy Range: {(max_accuracy - min_accuracy)*100:.2f}%")
