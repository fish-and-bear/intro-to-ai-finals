import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib
from sklearn.metrics import precision_score, recall_score, f1_score

# Set the font to a Chinese-compatible font
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# Load the dataset
file_path = 'translated_dataset.xlsx'
dataset = pd.read_excel(file_path, sheet_name='Translated Data')

# Data Preprocessing
dataset['Number of Deaths (死亡人數)'] = dataset['Number of Deaths (死亡人數)'].str.replace(',', '').astype(float)
dataset['Cause of Death Encoded'] = LabelEncoder().fit_transform(dataset['Cause of Death (疾病死因)'])

# One-hot encoding for the gender column
gender_encoder = OneHotEncoder()
encoded_gender = gender_encoder.fit_transform(dataset[['Gender (性別)']]).toarray()
gender_columns = gender_encoder.get_feature_names_out()
dataset = dataset.join(pd.DataFrame(encoded_gender, columns=gender_columns))

# Prepare data for training
feature_columns = ['Number of Deaths (死亡人數)', 'Death Rate (死亡率)', 'Standardized Death Rate (標準化死亡率)', 'Proportion of Deaths (死亡人數結構比)'] + list(gender_columns)
X = dataset[feature_columns]
y = dataset['Cause of Death Encoded']

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42)
}

# Hyperparameter grids
dt_params = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_params = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt'],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.2],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

knn_params = {
    'n_neighbors': [3, 5, 11, 19],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

lr_params = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

# Hyperparameters
params = {
    'Decision Tree': dt_params,
    'Random Forest': rf_params,
    'Gradient Boosting': gb_params,
    'K-Nearest Neighbors': knn_params,
    'Logistic Regression': lr_params
}

# Dictionary to store the best parameters for each model
best_params = {}

# Perform Grid Search or Randomized Search
for model_name, model in models.items():
    print(f"Running Search for {model_name}...")
    if model_name == "Gradient Boosting":
        search = RandomizedSearchCV(model, gb_params, n_iter=10, cv=3, n_jobs=-1, verbose=2, random_state=42)
    else:
        search = GridSearchCV(model, params[model_name], cv=3, n_jobs=-1, verbose=2)
    search.fit(X_train_scaled, y_train_scaled)
    best_params[model_name] = search.best_params_
    print(f"Best Parameters for {model_name}: {search.best_params_}")

# Train and evaluate models with the best parameters
results = {}
best_model = None
best_accuracy = 0

# Dictionary to store evaluation metrics for each model
model_metrics = {model_name: {} for model_name in models.keys()}

# Train and evaluate models with the best parameters
for name, model in models.items():
    print(f"Training {name} with best parameters...")
    model.set_params(**best_params[name])
    
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train_scaled)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    confusion = confusion_matrix(y_test, y_pred)
    results[name] = {'Accuracy': accuracy, 'Report': report, 'Confusion Matrix': confusion}

    # Correct variable name usage here
    model_metrics[name]['Accuracy'] = accuracy
    model_metrics[name]['Precision'] = report['weighted avg']['precision']
    model_metrics[name]['Recall'] = report['weighted avg']['recall']
    model_metrics[name]['F1-Score'] = report['weighted avg']['f1-score']

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = name

# Sort models by accuracy
sorted_models = dict(sorted(results.items(), key=lambda item: item[1]['Accuracy'], reverse=True))

# Visualizations
plt.rcParams['font.sans-serif'] = ['Arial', 'KaiTi', 'SimHei', 'FangSong']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.unicode_minus'] = False

# Visualization 1: Model Comparison (Sorted by Accuracy)
accuracies = [sorted_models[model]['Accuracy'] for model in sorted_models]
model_names = list(sorted_models.keys())

plt.figure(figsize=(12, 8))
sns.barplot(x=accuracies, y=model_names, hue=model_names, palette='viridis', dodge=False)
plt.title('Model Comparison - Accuracy (Sorted)')
plt.xlabel('Accuracy')
plt.ylabel('Model')
plt.xlim(0, 1)
plt.legend(loc='upper right', title='Model', bbox_to_anchor=(1.1, 1))
plt.show()

# Visualization 2: Confusion Matrix for the Best Model
class_labels = dataset['Cause of Death (疾病死因)'].unique()
best_model_confusion = results[best_model]['Confusion Matrix']

plt.figure(figsize=(60, 60))
sns.heatmap(best_model_confusion, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title(f'Confusion Matrix for {best_model}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()

# Output results for each model
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    print(f"Accuracy: {metrics['Accuracy']}")
    print(metrics['Report'])

# Displaying best model
print(f"Best Model: {best_model} with Accuracy: {best_accuracy}")

# Convert the metrics to a DataFrame for easy plotting
metrics_df = pd.DataFrame(model_metrics).T

# Plotting the metrics
plt.figure(figsize=(10, 6))
metrics_df.plot(kind='bar', ax=plt.gca())
plt.title('Model Comparison using Various Metrics')
plt.ylabel('Scores')
plt.xlabel('Models')
plt.xticks(rotation=45)
plt.legend(title='Metrics')
plt.tight_layout()
plt.show()