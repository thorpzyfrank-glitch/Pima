import pandas as pd
import numpy as np
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionary to store all models
models = {}
accuracies = {}

# 1. Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
models['Logistic Regression'] = lr
accuracies['Logistic Regression'] = lr_accuracy
print(f"1. Logistic Regression: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")

# 2. Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42, max_depth=10)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
models['Decision Tree'] = dt
accuracies['Decision Tree'] = dt_accuracy
print(f"2. Decision Tree: {dt_accuracy:.4f} ({dt_accuracy*100:.2f}%)")

# 3. Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
models['Random Forest'] = rf
accuracies['Random Forest'] = rf_accuracy
print(f"3. Random Forest: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")

# 4. Support Vector Machine (SVM)
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
models['SVM'] = svm
accuracies['SVM'] = svm_accuracy
print(f"4. Support Vector Machine: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")

# 5. K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
models['K-Nearest Neighbors'] = knn
accuracies['K-Nearest Neighbors'] = knn_accuracy
print(f"5. K-Nearest Neighbors: {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)")

# 6. Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)
models['Naive Bayes'] = nb
accuracies['Naive Bayes'] = nb_accuracy
print(f"6. Naive Bayes: {nb_accuracy:.4f} ({nb_accuracy*100:.2f}%)")

# 7. Gradient Boosting Classifier
gb = GradientBoostingClassifier(random_state=42, max_depth=5)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
models['Gradient Boosting'] = gb
accuracies['Gradient Boosting'] = gb_accuracy
print(f"7. Gradient Boosting: {gb_accuracy:.4f} ({gb_accuracy*100:.2f}%)")

# 8. AdaBoost Classifier
ab = AdaBoostClassifier(random_state=42, n_estimators=50)
ab.fit(X_train, y_train)
ab_pred = ab.predict(X_test)
ab_accuracy = accuracy_score(y_test, ab_pred)
models['AdaBoost'] = ab
accuracies['AdaBoost'] = ab_accuracy
print(f"8. AdaBoost: {ab_accuracy:.4f} ({ab_accuracy*100:.2f}%)")

# 9. Neural Network (MLPClassifier)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_pred)
models['Neural Network'] = mlp
accuracies['Neural Network'] = mlp_accuracy
print(f"9. Neural Network (MLP): {mlp_accuracy:.4f} ({mlp_accuracy*100:.2f}%)")

# 10. Voting Classifier (Ensemble)
voting = VotingClassifier(
    estimators=[('lr', LogisticRegression(max_iter=1000, random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('svm', SVC(kernel='rbf', random_state=42, probability=True))],
    voting='soft'
)
voting.fit(X_train, y_train)
voting_pred = voting.predict(X_test)
voting_accuracy = accuracy_score(y_test, voting_pred)
models['Voting Classifier'] = voting
accuracies['Voting Classifier'] = voting_accuracy
print(f"10. Voting Classifier: {voting_accuracy:.4f} ({voting_accuracy*100:.2f}%)")

# Save accuracy scores (avoiding pickling complex models)
with open('accuracies.pkl', 'wb') as f:
    pickle.dump(accuracies, f)

print("\n" + "="*50)
print("All models trained and saved successfully!")
print("="*50)

# Find best model
best_model = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_model]
print(f"\n🏆 Best Model: {best_model}")
print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# Sort and display all models by accuracy
print("\n" + "="*50)
print("Model Rankings by Accuracy:")
print("="*50)
sorted_models = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
for i, (model_name, accuracy) in enumerate(sorted_models, 1):
    print(f"{i}. {model_name:.<30} {accuracy:.4f} ({accuracy*100:.2f}%)")
