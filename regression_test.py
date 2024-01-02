import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib
import seaborn as sns

# Set the font to a Chinese-compatible font, e.g., SimHei
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] # Use SimHei font
matplotlib.rcParams['axes.unicode_minus'] = False # Solve the minus sign display issue


# Load the dataset
dataset = pd.read_excel('translated_dataset.xlsx', sheet_name='Translated Data')

# Preprocessing
dataset['Year (年份)'] = dataset['Year (年份)'].apply(lambda x: int(x.split(' ')[0]))

# Features and target
features = ['Year (年份)', 'Gender (性別)', 'Cause of Death (疾病死因)']
target = 'Proportion of Deaths (死亡人數結構比)'

# OneHotEncoder for categorical features
categorical_transformer = OneHotEncoder()

# Column Transformer
preprocessor = ColumnTransformer(
    transformers=[('cat', categorical_transformer, ['Gender (性別)', 'Cause of Death (疾病死因)'])],
    remainder='passthrough'
)

# Gradient Boosting Regressor with optimized parameters
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1))
])

# Train-test split
X = dataset[features]
y = dataset[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
pipeline.fit(X_train, y_train)

# Synthetic data for future predictions
unique_genders = dataset['Gender (性別)'].unique()
unique_causes = dataset['Cause of Death (疾病死因)'].unique()
synthetic_data = pd.DataFrame({
    'Year (年份)': np.repeat(np.arange(2024, 2031), len(unique_genders) * len(unique_causes)),
    'Gender (性別)': np.tile(np.repeat(unique_genders, len(unique_causes)), 7),
    'Cause of Death (疾病死因)': np.tile(unique_causes, len(unique_genders) * 7)
})

# Predictions
predicted_proportions = pipeline.predict(synthetic_data)
synthetic_data['Predicted Proportion of Deaths'] = predicted_proportions

# Generating conclusion statements
synthetic_data['Conclusion'] = synthetic_data.apply(
    lambda row: f"For the year {row['Year (年份)']}, there will be a {row['Predicted Proportion of Deaths']:.2f}% "
                f"proportion of deaths for {row['Gender (性別)']} gender due to {row['Cause of Death (疾病死因)']}.",
    axis=1
)

# Displaying predictions in table form
table = synthetic_data[['Year (年份)', 'Gender (性別)', 'Cause of Death (疾病死因)', 'Predicted Proportion of Deaths']]
print(table)

# Visualization - Bar Chart for a specific year
specific_year = 2024
#specific_gender = 'Male (男性)'
specific_gender = 'Female (女性)'
#specific_gender = 'Total (總計)'
year_data = synthetic_data[(synthetic_data['Year (年份)'] == specific_year) & (synthetic_data['Gender (性別)'] == specific_gender)]

# Grouping data by 'Cause of Death' and summing the predicted proportions
grouped_data = year_data.groupby('Cause of Death (疾病死因)', as_index=False)['Predicted Proportion of Deaths'].sum()

# Exclude 'All Causes of Death' from the data
grouped_data = grouped_data[grouped_data['Cause of Death (疾病死因)'] != 'All Causes of Death (所有死因)']

# Sorting data from highest to lowest
grouped_data = grouped_data.sort_values(by='Predicted Proportion of Deaths', ascending=False)

# Creating the bar chart
plt.figure(figsize=(20, 10))
sns.barplot(x='Cause of Death (疾病死因)', y='Predicted Proportion of Deaths', data=grouped_data, 
            palette='viridis', hue='Cause of Death (疾病死因)', dodge=False)
plt.title(f'Predicted Death Proportions by Cause of Death in {specific_year} for {specific_gender}')
plt.ylabel('Predicted Proportion of Deaths')
plt.xlabel('Cause of Death')
plt.xticks(rotation=90)
plt.show()

