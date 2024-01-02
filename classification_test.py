import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
excel_file_path = 'translated_dataset.xlsx'
translated_data_df = pd.read_excel(excel_file_path, sheet_name='Translated Data')

# Data Preprocessing
# Converting 'Year' to numeric value by extracting the first 4 characters
translated_data_df['Year'] = translated_data_df['Year (年份)'].str[:4].astype(int)

# Encoding the 'Gender' categorical variable
gender_encoder = LabelEncoder()
translated_data_df['Gender_Encoded'] = gender_encoder.fit_transform(translated_data_df['Gender (性別)'])

# Selecting relevant columns
features = ['Year', 'Gender_Encoded']
target = 'Cause of Death (疾病死因)'

# Splitting the data
X = translated_data_df[features]
y = translated_data_df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Training
rf_model = RandomForestClassifier(max_depth=None, max_features='sqrt', min_samples_leaf=1, min_samples_split=5, n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Function to make new predictions
def predict_cause_of_death(year, gender):
    # Encoding the gender
    gender_encoded = gender_encoder.transform([gender])
    # Creating DataFrame for the input
    input_data = pd.DataFrame({'Year': [year], 'Gender_Encoded': gender_encoded})
    # Making prediction
    prediction = rf_model.predict(input_data)
    return prediction[0]

# Example usage
year = 2024
gender = 'Male (男性)'
example_prediction = predict_cause_of_death(year, gender)
print("Most likely cause of death for", gender, "in Taipei in the year", year, ":", example_prediction)

year = 2050
gender = 'Total (總計)'
example_prediction = predict_cause_of_death(year, gender)
print("Most likely cause of death for", gender, "in Taipei in the year", year, ":", example_prediction)
