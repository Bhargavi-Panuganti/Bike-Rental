import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

# Title of the app
st.title('Bike Rental Prediction')

# Function to make predictions
def predict_bike_rentals(season, holiday, workingday, weather, temp, atemp, humidity, windspeed):
    # Input features should match the training columns
    features = np.array([[season, holiday, workingday, weather, temp, atemp, humidity, windspeed]])
    prediction = model.predict(features)
    return int(prediction)

# Create input fields for the user to provide feature values
season = st.selectbox('Season', [1, 2, 3, 4])  # Assumes seasons are encoded as 1-4
holiday = st.selectbox('Holiday', [0, 1])  # 0 = No holiday, 1 = Holiday
workingday = st.selectbox('Working Day', [0, 1])  # 0 = No, 1 = Yes
weather = st.selectbox('Weather', [1, 2, 3, 4])  # Assumes weather conditions encoded as 1-4
temp = st.slider('Temperature', min_value=-10.0, max_value=40.0, value=20.0)  # In Celsius
atemp = st.slider('Feels Like Temperature', min_value=-10.0, max_value=50.0, value=25.0)  # In Celsius
humidity = st.slider('Humidity (%)', min_value=0, max_value=100, value=50)
windspeed = st.slider('Windspeed', min_value=0.0, max_value=50.0, value=10.0)


# Button to trigger prediction
if st.button('Predict'):
    result = predict_bike_rentals(season, holiday, workingday, weather, temp, atemp, humidity, windspeed)
    st.success(f'Predicted number of bike rentals: {result}')
