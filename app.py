import streamlit as st
from models.neural_network_model import load_model  # Import the load_model function
import pandas as pd

# Load the trained model (no need for create_model here)
model = load_model()  # This will load the pre-trained model

# Streamlit app UI
st.title("Salary Prediction Using Neural Network")

# Inputs for the user
experience = st.number_input("Enter Experience (years):", min_value=0, max_value=50, value=0)
education_level = st.number_input("Enter Education Level (1-5):", min_value=1, max_value=5, value=1)
age = st.number_input("Enter Age:", min_value=18, max_value=100, value=22)

# Predict the salary when inputs are entered
if st.button("Predict Salary"):
    # Prepare the input data
    input_data = pd.DataFrame([[experience, education_level, age]], columns=["Experience", "Education_Level", "Age"])
    
    # Make prediction
    predicted_salary = model.predict(input_data)[0]
    
    # Show the prediction
    st.write(f"Predicted Salary: ${predicted_salary:,.2f}")
