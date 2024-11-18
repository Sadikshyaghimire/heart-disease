import pandas as pd
import numpy as np
import warnings
import pickle

# Settings
pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings('ignore')

# Load data
try:
    data = pd.read_csv('heart.csv')
except FileNotFoundError:
    raise FileNotFoundError("The dataset 'heart.csv' was not found.")

# Display initial data information
print(data.head())
print(f"Data Shape: {data.shape}")
print(f"Data Columns: {data.columns.tolist()}")
data.info()

# Handle missing values if necessary (uncomment if needed)
# data.fillna(method='ffill', inplace=True)  # Example of filling missing values with forward fill

# Descriptive statistics
print("\nDescriptive Statistics:")
print(data.describe().T)

# Splitting data for those with and without heart disease
yes = data[data['HeartDisease'] == 1].describe().T
no = data[data['HeartDisease'] == 0].describe().T

# Print out some stats for heart disease and no heart disease categories
print("\nStatistics for Heart Disease:")
print(yes[['mean']])

print("\nStatistics for No Heart Disease:")
print(no[['mean']])

# Identify categorical and numerical features
col = list(data.columns)
categorical_features = []
numerical_features = []

for column in col:
    if data[column].dtype == 'object':
        categorical_features.append(column)
    else:
        numerical_features.append(column)

print(f"\nCategorical Features: {categorical_features}")
print(f"Numerical Features: {numerical_features}")

# Encoding categorical features for further analysis
data_encoded = pd.get_dummies(data, drop_first=True)
print(f"\nEncoded Data Columns: {data_encoded.columns.tolist()}")

# Splitting the data into features and target variable
X = data_encoded.drop('HeartDisease', axis=1)
y = data_encoded['HeartDisease']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training using Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Initialize and fit the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model using pickle
with open('heart_disease_rf_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model saved to 'heart_disease_rf_model.pkl'")

# To load the model later, use this code:
# with open('heart_disease_rf_model.pkl', 'rb') as model_file:
#     loaded_model = pickle.load(model_file)
#     print("Model loaded successfully!")
