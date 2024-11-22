import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

def load_trained_model(model_path="./models/random_forest_model.pkl"):
    """
    Load the trained Random Forest model.
    
    Args:
    model_path (str): Path to the saved model file.
    
    Returns:
    sklearn.ensemble.RandomForestClassifier: Loaded model.
    """
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    return model

def load_label_encoder(encoder_path='./data/scaler/label_encoder.pkl'):
    """
    Load the LabelEncoder used for encoding the 'RiskLevel' labels.
    
    Args:
    encoder_path (str): Path to the saved label encoder.
    
    Returns:
    LabelEncoder: Loaded LabelEncoder.
    """
    with open(encoder_path, 'rb') as encoder_file:
        label_encoder = pickle.load(encoder_file)
    return label_encoder

def load_scaler(scaler_path="./data/scaler/scaler.pkl"):
    """
    Load the saved scaler for preprocessing.
    
    Args:
    scaler_path (str): Path to the saved scaler file.
    
    Returns:
    sklearn.preprocessing.StandardScaler: Loaded scaler.
    """
    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler

def predict(model, scaler, label_encoder, new_data):
    """
    Make predictions using the trained model and preprocess the input data.
    
    Args:
    model (sklearn model): Trained model.
    scaler (sklearn.preprocessing.StandardScaler): Scaler for input features.
    label_encoder (sklearn.preprocessing.LabelEncoder): Label encoder for decoding predictions.
    new_data (pd.DataFrame): New data for prediction.
    
    Returns:
    list: Predicted class labels.
    """
    new_data_scaled = scaler.transform(new_data)
    predictions = model.predict(new_data_scaled)
    
    # Ensure predictions are integers
    predictions = predictions.astype(int)
    
    # Decode predictions using LabelEncoder
    decoded_predictions = label_encoder.inverse_transform(predictions)
    return decoded_predictions

if __name__ == "__main__":
    # Load the model, scaler, and label encoder
    model = load_trained_model()
    label_encoder = load_label_encoder()
    scaler = load_scaler()

    # Example data for prediction (adjust the feature values as needed)
    example_data = pd.DataFrame({
        "Age": [25],
        "SystolicBP": [130],
        "DiastolicBP": [80],
        "BS": [15.00],
        "BodyTemp": [98.0],
        "HeartRate": [86],
    })

    # Make predictions
    predictions = predict(model, scaler, label_encoder, example_data)
    
    print("Prediction (Risk Level):", predictions[0])
