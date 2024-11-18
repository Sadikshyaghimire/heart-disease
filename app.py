import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model from pickle
with open('heart_disease_rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Column names the model expects (These should be the same columns used during training)
expected_columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Function to make predictions
def predict_heart_disease(input_data):
    # Ensure the input data has the same features as expected by the model
    input_df = pd.DataFrame([input_data], columns=expected_columns)
    
    # One-hot encode the categorical columns (if necessary)
    input_df_encoded = pd.get_dummies(input_df, drop_first=True)

    # Ensure the same columns are present after encoding
    # Add any missing columns by using the same columns from the training model
    missing_cols = set(model.feature_names_in_) - set(input_df_encoded.columns)
    for col in missing_cols:
        input_df_encoded[col] = 0  # Add missing columns with default value 0

    # Reorder the columns to match the model's training set
    input_df_encoded = input_df_encoded[model.feature_names_in_]

    # Make the prediction
    prediction = model.predict(input_df_encoded)
    return prediction[0]

# Streamlit UI
st.title("Heart Disease Prediction Dashboard")

# Input fields for the features (adjust these based on the columns in your dataset)
age = st.slider('Age', min_value=29, max_value=77, value=50)
sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3], format_func=lambda x: f'Type {x}')
trestbps = st.slider('Resting Blood Pressure (mm Hg)', min_value=94, max_value=200, value=120)
chol = st.slider('Serum Cholesterol (mg/dl)', min_value=126, max_value=564, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
restecg = st.selectbox('Resting Electrocardiographic Results', options=[0, 1, 2], format_func=lambda x: f'Result {x}')
thalach = st.slider('Maximum Heart Rate Achieved', min_value=71, max_value=202, value=150)
exang = st.selectbox('Exercise Induced Angina', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
oldpeak = st.slider('Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=6.2, value=1.0)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=[0, 1, 2], format_func=lambda x: f'Slope {x}')
ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', options=[0, 1, 2, 3], format_func=lambda x: f'Vessel {x}')
thal = st.selectbox('Thalassemia', options=[0, 1, 2], format_func=lambda x: f'Thal {x}')

# Convert inputs into a list for prediction
input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])

# Make prediction when button is clicked
if st.button('Predict'):
    result = predict_heart_disease(input_data)
    
    if result == 1:
        st.error("The person has heart disease.")
    else:
        st.success("The person does not have heart disease.")
