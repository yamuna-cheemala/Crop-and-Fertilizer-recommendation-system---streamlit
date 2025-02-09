import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Add Streamlit page config to set the title and icon
st.set_page_config(page_title="Crop and Fertilizer Recommendation System", page_icon="🌾")

# Project Header
st.title("Crop and Fertilizer Recommendation System")
st.write("""
    Welcome to the Crop and Fertilizer Recommendation System! 
    This tool helps farmers make informed decisions by recommending the best crops to grow based on soil conditions, 
    and the most suitable fertilizers based on crop and environmental factors.
""")

# Fertilizer Prediction System

# Load the dataset for fertilizer prediction
fertilizer = pd.read_csv("dataset/Fertilizer Prediction.csv")

# Initialize the label encoders for Soil Type and Crop Type
label_encoder_soil = LabelEncoder()
label_encoder_crop = LabelEncoder()

# Fit label encoders on the entire unique values of the Soil Type and Crop Type columns
label_encoder_soil.fit(fertilizer['Soil Type'].unique())
label_encoder_crop.fit(fertilizer['Crop Type'].unique())

# Create fertilizer number column
fertilizer['fert_no'] = fertilizer['Fertilizer Name'].map({
    'Urea': 1, 'DAP': 2, '14-35-14': 3, '28-28': 4, '17-17-17': 5, '20-20': 6, '10-26-26': 7
})
fertilizer.drop('Fertilizer Name', axis=1, inplace=True)

# Encode the categorical variables 'Soil Type' and 'Crop Type'
fertilizer['Soil Type'] = label_encoder_soil.transform(fertilizer['Soil Type'])
fertilizer['Crop Type'] = label_encoder_crop.transform(fertilizer['Crop Type'])

# Define features (X) and target (y) for fertilizer prediction
X_fertilizer = fertilizer.drop('fert_no', axis=1)
y_fertilizer = fertilizer['fert_no']

# Train-test split for fertilizer model
X_train_fertilizer, X_test_fertilizer, y_train_fertilizer, y_test_fertilizer = train_test_split(X_fertilizer, y_fertilizer, test_size=0.2, random_state=42)

# Scale the input features
scaler_fertilizer = StandardScaler()
X_train_fertilizer = scaler_fertilizer.fit_transform(X_train_fertilizer)
X_test_fertilizer = scaler_fertilizer.transform(X_test_fertilizer)

# Train the Decision Tree Classifier for fertilizer prediction
fertilizer_model = DecisionTreeClassifier(random_state=42)
fertilizer_model.fit(X_train_fertilizer, y_train_fertilizer)

# Fertilizer Recommendation Section
st.header("Fertilizer Recommendation ")

st.subheader(" Find the best fertilizer for your crops")

# Get unique soil and crop types from the dataset to provide valid options
unique_soil_types = fertilizer['Soil Type'].unique()
unique_crop_types = fertilizer['Crop Type'].unique()

# Map the encoded values back to their original labels
soil_type_options = label_encoder_soil.inverse_transform(unique_soil_types)
crop_type_options = label_encoder_crop.inverse_transform(unique_crop_types)

# Input fields for fertilizer recommendation
temperature_fertilizer = st.number_input("Enter Temperature (°C) for Fertilizer", min_value=-50, max_value=50, step=1)
humidity_fertilizer = st.number_input("Enter Humidity (%) for Fertilizer", min_value=0, max_value=100, step=1)
moisture = st.number_input("Enter Moisture Content (%)", min_value=0, max_value=100, step=1)
soil_type = st.selectbox("Select Soil Type", options=soil_type_options)
crop_type = st.selectbox("Select Crop Type", options=crop_type_options)
nitrogen = st.number_input("Enter Nitrogen (N) for Fertilizer", min_value=0, max_value=300, step=1)
potassium = st.number_input("Enter Potassium (K) for Fertilizer", min_value=0, max_value=300, step=1)
phosphorous = st.number_input("Enter Phosphorous (P) for Fertilizer", min_value=0, max_value=300, step=1)

if st.button('Recommend Fertilizer'):
    try:
        # Encode soil type and crop type
        soil_type_encoded = label_encoder_soil.transform([soil_type])[0]
        crop_type_encoded = label_encoder_crop.transform([crop_type])[0]

        # Prepare the input data for prediction (numeric values + encoded categorical values)
        fertilizer_input = np.array([[temperature_fertilizer, humidity_fertilizer, moisture, soil_type_encoded, crop_type_encoded, nitrogen, potassium, phosphorous]])

        # Apply scaling to the input data
        transformed_input_fertilizer = scaler_fertilizer.transform(fertilizer_input)

        # Make the fertilizer prediction
        fertilizer_prediction = fertilizer_model.predict(transformed_input_fertilizer)

        # Map the predicted fertilizer number back to its name
        fert_result = {1: 'Urea', 2: 'DAP', 3: '14-35-14', 4: '28-28', 5: '17-17-17', 6: '20-20', 7: '10-26-26'}
        st.write(f"The best fertilizer to use is: **{fert_result[fertilizer_prediction[0]]}**")
    
    except Exception as e:
        st.error(f" Error: {e}")

# Crop Prediction System

# Load the crop recommendation dataset
crop = pd.read_csv("dataset/Crop_recommendation.csv")

# Preprocess the crop dataset
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6, 'orange': 7,
    'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11, 'mango': 12, 'banana': 13,
    'pomegranate': 14, 'lentil': 15, 'blackgram': 16, 'mungbean': 17, 'mothbeans': 18,
    'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}
crop['crop_no'] = crop['label'].map(crop_dict)
crop.drop('label', axis=1, inplace=True)

# Prepare the feature and target columns for the crop model
X_crop = crop.drop('crop_no', axis=1)
y_crop = crop['crop_no']

# Train-test split for crop model
X_train_crop, X_test_crop, y_train_crop, y_test_crop = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)

# Scale the input features
scaler_crop = StandardScaler()
X_train_crop = scaler_crop.fit_transform(X_train_crop)
X_test_crop = scaler_crop.transform(X_test_crop)

# Train the Decision Tree Classifier for crop prediction
crop_model = DecisionTreeClassifier(random_state=42)
crop_model.fit(X_train_crop, y_train_crop)

# Crop Recommendation Section
st.header("Crop Recommendation ")

st.subheader("Find the best crop for your soil conditions")

# Input fields for crop recommendation
nitrogen_crop = st.number_input("Enter Nitrogen (N) for Crop", min_value=0, max_value=300, step=1)
phosphorous_crop = st.number_input("Enter Phosphorous (P) for Crop", min_value=0, max_value=300, step=1)
potassium_crop = st.number_input("Enter Potassium (K) for Crop", min_value=0, max_value=300, step=1)
temperature_crop = st.number_input("Enter Temperature (°C) for Crop", min_value=-50, max_value=50, step=1)
humidity_crop = st.number_input("Enter Humidity (%) for Crop", min_value=0, max_value=100, step=1)
ph_crop = st.number_input("Enter pH for Crop", min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.number_input("Enter Rainfall (mm) for Crop", min_value=0, max_value=1000, step=1)

if st.button('Recommend Crop'):
    try:
        # Prepare the input data for prediction (numeric values)
        crop_input = np.array([[nitrogen_crop, phosphorous_crop, potassium_crop, temperature_crop, humidity_crop, ph_crop, rainfall]])

        # Apply scaling to the input data
        transformed_input_crop = scaler_crop.transform(crop_input)

        # Make the crop prediction
        crop_prediction = crop_model.predict(transformed_input_crop)

        # Map the predicted crop number back to its name
        crop_result = {1: 'Rice', 2: 'Maize', 3: 'Jute', 4: 'Cotton', 5: 'Coconut', 6: 'Papaya', 7: 'Orange',
                       8: 'Apple', 9: 'Muskmelon', 10: 'Watermelon', 11: 'Grapes', 12: 'Mango', 13: 'Banana',
                       14: 'Pomegranate', 15: 'Lentil', 16: 'Blackgram', 17: 'Mungbean', 18: 'Mothbeans', 
                       19: 'Pigeonpeas', 20: 'Kidneybeans', 21: 'Chickpea', 22: 'Coffee'}

        # Show the result
        st.write(f"The best crop to grow is: **{crop_result[crop_prediction[0]]}**")

    except Exception as e:
        st.error(f" Error: {e}")
