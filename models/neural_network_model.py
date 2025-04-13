from sklearn.neural_network import MLPRegressor
import joblib
import pandas as pd
import numpy as np

# Function to train a simple MLP model
def create_model(X_train, y_train):
    model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=10000, random_state=1)
    model.fit(X_train, y_train)
    return model

# Function to save the trained model
def save_model(model, filename="models/neural_network_model.pkl"):
    joblib.dump(model, filename)

# Function to load the trained model
def load_model(filename="models/neural_network_model.pkl"):
    return joblib.load(filename)

# Sample dataset (same as before, just for example)
def get_sample_data():
    data = {
        "Experience": [1, 2, 3, 4, 5],
        "Education_Level": [1, 2, 3, 4, 5],
        "Age": [22, 25, 28, 30, 35],
        "Salary": [40000, 45000, 50000, 55000, 60000]
    }
    df = pd.DataFrame(data)
    X = df[["Experience", "Education_Level", "Age"]]
    y = df["Salary"]
    return X, y

# Train and save the model
X, y = get_sample_data()
model = create_model(X, y)
save_model(model)
