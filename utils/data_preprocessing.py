from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # Example preprocessing (e.g., standardizing numerical features)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

