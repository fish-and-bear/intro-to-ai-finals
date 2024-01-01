import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import RFECV
import shap
import xgboost as xgb
import catboost as cb
import ipywidgets as widgets
widgets.IntSlider()

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Load and inspect the dataset
file_path = 'translated_dataset.xlsx'
df = pd.read_excel(file_path, sheet_name='Translated Data')

# Data Inspection
print(df.head())
print(df.info())
print(df.describe())

# Splitting the data into features (X) and target (y)
X = df.drop(columns=['Proportion of Deaths (死亡人數結構比)'])
y = df['Proportion of Deaths (死亡人數結構比)']

# Identifying categorical and numerical columns for preprocessing
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Creating a preprocessing pipeline with Polynomial Features
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Define various regression models for comparison
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(),
    'Gradient Boosting Regressor': GradientBoostingRegressor(),
    'SVR': SVR(),
    'K-Neighbors Regressor': KNeighborsRegressor(),
    'Lasso Regression': Lasso(),
    'XGBoost': xgb.XGBRegressor(),
    'CatBoost': cb.CatBoostRegressor(verbose=0)
}

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning with RandomizedSearchCV
param_distributions = {
    'Random Forest Regressor': {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [10, 20, None],
        'model__min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting Regressor': {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7]
    },
    'XGBoost': {
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__max_depth': [3, 5, 7]
    },
    'CatBoost': {
        'model__iterations': [100, 200, 300],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__depth': [4, 6, 8]
    }
}

# Perform hyperparameter tuning
for model_name in ['Random Forest Regressor', 'Gradient Boosting Regressor', 'XGBoost', 'CatBoost']:
    model = models[model_name]
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    random_search = RandomizedSearchCV(pipeline, param_distributions[model_name], n_iter=10, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    print(f"Best parameters for {model_name}: {best_params}")
    models[model_name].set_params(**{k.split('__')[1]: v for k, v in best_params.items()})

# Dictionary to store model performances
model_performance = {}

# Training and evaluating each model with the best parameters
for model_name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    model_performance[model_name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2 Score': r2}

    # Scatter Plot for each model
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Ideal prediction line
    plt.title(f'Actual vs Predicted - {model_name}', fontsize=14)
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.show()

# Consolidated Metric Comparison Chart
metrics_df = pd.DataFrame(model_performance).T
metrics_df = metrics_df.reset_index().melt(id_vars='index')
metrics_df.rename(columns={'index': 'Model', 'variable': 'Metric', 'value': 'Score'}, inplace=True)

plt.figure(figsize=(12, 8))
sns.barplot(x='Metric', y='Score', hue='Model', data=metrics_df)
plt.title('Comparison of Models Across Different Metrics', fontsize=16)
plt.xlabel('Metric', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.legend(title='Model')
plt.show()


# Training and evaluating each model with the best parameters
for model_name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    model_performance[model_name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2 Score': r2}

# Model Ranking based on RMSE
rmse_scores = {model: metrics['RMSE'] for model, metrics in model_performance.items()}
best_model_name = min(rmse_scores, key=rmse_scores.get)
best_model = models[best_model_name]

# Dictionary to store model performances
model_performance = {}

# Training and evaluating each model with the best parameters
for model_name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    model_performance[model_name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2 Score': r2}

# Model Ranking based on RMSE
rmse_scores = {model: metrics['RMSE'] for model, metrics in model_performance.items()}
sorted_models = sorted(rmse_scores.items(), key=lambda x: x[1])

# Plotting RMSE Rankings
plt.figure(figsize=(10, 6))
sns.barplot(x=[model for model, _ in sorted_models], y=[rmse for _, rmse in sorted_models])
plt.title('Model Ranking based on RMSE', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.xticks(rotation=45)
plt.show()

# Printing the ranking table
print("Model Ranking based on RMSE (Lower is better):")
ranking_table = pd.DataFrame(sorted_models, columns=['Model', 'RMSE']).reset_index(drop=True)
print(ranking_table)

# Identifying the best model
best_model_name, best_rmse = sorted_models[0]
print(f"\nThe best model is {best_model_name} with an RMSE of {best_rmse:.3f}")

# Explanation for why the best model is the best
print("\nWhy the Best Model is the Best:")
print(f"The {best_model_name} has the lowest RMSE value among all the models tested, indicating it has the smallest average error in its predictions. This metric suggests that it's the most accurate model for this dataset.")