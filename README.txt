Salary Prediction App using Neural Network

This is a simple web application built using Streamlit that predicts a person's salary based on their years of experience, education level, and age. The prediction is powered by a neural network model trained using TensorFlow/Keras.

---

Features:
- User-friendly Streamlit interface
- Inputs: Experience, Education Level (1–5), Age
- Real-time salary prediction
- Modular and clean code (or single-file version)
- Includes model training and saving capabilities

---

Project Contents:
- app.py / salary_prediction_app.py – Main Streamlit app (may include training logic)
- data/dataset.csv – Sample dataset for training
- models/neural_network_model.h5 – Trained Keras model (saved after training)
- requirements.txt – Dependencies list

---

How to Run:

1. Install Python dependencies:

   pip install -r requirements.txt

2. If a trained model is not present, it will be trained automatically when the app is first run.

3. To launch the app:

   streamlit run app.py

---

Model Info:
- Input features: Experience, Education Level, Age
- Output: Predicted Salary
- Neural network with two hidden layers
- Loss function: Mean Squared Error
- Optimizer: Adam

---

Dataset Format (CSV):
The dataset should have the following columns:

Experience, Education_Level, Age, Salary

Example:

3,2,28,50000  
5,4,32,72000  
...

---

Example Usage:
Experience: 5  
Education Level: 3  
Age: 30  
→ Predicted Salary: $72,400.00

---

Requirements:
- streamlit
- pandas
- numpy
- scikit-learn
- tensorflow

Make sure these are listed in `requirements.txt`.

---
Link: 
https://neural-ager7nyxsswt4c6ugsvirl.streamlit.app/  

Notes:
- You can modify `train_model.py` or the training logic in the main script to adjust model parameters.
- If no model exists, one will be trained using `data/dataset.csv`.

---

Contact:
Eduardo Queija de Siqueira

---


