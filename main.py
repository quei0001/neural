import pandas as pd
from sklearn.model_selection import train_test_split
from models.neural_network_model import create_model  # Import model creation function
from utils.data_loader import load_data
from utils.data_preprocessing import preprocess_data
import numpy as np

def main():
    # Step 1: Load the dataset
    dataset = load_data('data/dataset.csv')
    print(f"Dataset loaded with {dataset.shape[0]} rows and {dataset.shape[1]} columns.")
    
    # Step 2: Preprocess the data
    X = dataset[['Experience', 'Education_Level', 'Age']].values
    y = dataset['Salary'].values

    # Normalize the input features
    X = preprocess_data(X)
    
    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 4: Create the model (this function comes from `neural_network_model.py`)
    model = create_model(input_shape=(X_train.shape[1],))
    
    # Step 5: Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    print("Model training complete.")

    # Step 6: Save the model as .h5 (optional, for future use)
    model.save('models/neural_network_model.h5')
    
    # Step 7: Evaluate the model on test data
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    
if __name__ == "__main__":
    main()
