import os
import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle

#Loading and inspecting the dataset
def load_and_inspect_data(file_path):
    # Loading the dataset
    df = pd.read_csv(file_path)
    
    # Displaying basic information
    print("Dataset basic information:")
    print(df.info())
    
    # Detailed statistics
    print("\nSummary Detailed statistics:")
    print(df.describe())
    
    return df

# Encoding categorical column
def encode_categorical_column(df, encoder_dir='./data/scaler'):
    if not os.path.exists(encoder_dir):
        os.makedirs(encoder_dir)
    
    encoder_path = os.path.join(encoder_dir, 'label_encoder.pkl')
    
    if not os.path.exists(encoder_path):
        label_encoder = LabelEncoder()
        df['RiskLevel'] = label_encoder.fit_transform(df['RiskLevel'])
        
        # Save LabelEncoder
        with open(encoder_path, 'wb') as encoder_file:
            pickle.dump(label_encoder, encoder_file)
        print(f"LabelEncoder saved to: {encoder_path}")
    else:
        # Load existing LabelEncoder
        with open(encoder_path, 'rb') as encoder_file:
            label_encoder = pickle.load(encoder_file)
        df['RiskLevel'] = label_encoder.transform(df['RiskLevel'])
        print(f"Loaded existing LabelEncoder from: {encoder_path}")
    
    return df, label_encoder



# Handling outliers
def cap_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_no_outliers = df.copy()
    
    for column in df.columns:
        lower_bound = Q1[column] - 1.5 * IQR[column]
        upper_bound = Q3[column] + 1.5 * IQR[column]
        df_no_outliers[column] = np.where(df[column] < lower_bound, lower_bound, df_no_outliers[column])
        df_no_outliers[column] = np.where(df[column] > upper_bound, upper_bound, df_no_outliers[column])
    
    return df_no_outliers

# Splitting into train, Val, test sets
def split_data(df):
    X = df.drop('RiskLevel', axis=1)
    y = df['RiskLevel']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Handling class imbalance using SMOTE
def handle_imbalance(X_train, y_train):
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

# Scaling the features
def scale_features(X_train_resampled, X_val, X_test, scaler_dir='./data/scaler'):
    if not os.path.exists(scaler_dir):
        os.makedirs(scaler_dir)
    
    scaler_path = os.path.join(scaler_dir, 'scaler.pkl')
    
    if not os.path.exists(scaler_path):
        scaler = StandardScaler()
        X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        with open('scaler.pkl', 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)
    else:
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        X_train_resampled_scaled = scaler.transform(X_train_resampled)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
    return X_train_resampled_scaled, X_val_scaled, X_test_scaled

# Save training and test data
# Function to save training and test data to CSV
def save_data_to_csv(X_train_resampled_scaled, X_val_scaled, X_test_scaled, y_train_resampled, y_val, y_test,
                     data_dir='./data', train_dir='./data/train', test_dir='./data/test'):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Save feature data only if files dont exist
    if not os.path.exists(os.path.join(train_dir, 'X_train_scaled.csv')):
        pd.DataFrame(X_train_resampled_scaled).to_csv('X_train_scaled.csv', index=False)
    if not os.path.exists(os.path.join(test_dir, 'X_val_scaled.csv')):
        pd.DataFrame(X_val_scaled).to_csv('X_val_scaled.csv', index=False)
    if not os.path.exists(os.path.join(test_dir, 'X_test_scaled.csv')):
        pd.DataFrame(X_test_scaled).to_csv('X_test_scaled.csv', index=False)
    
    # Save target data
    if not os.path.exists(os.path.join(train_dir, 'y_train_resampled.csv')):
        pd.DataFrame(y_train_resampled).to_csv('y_train_resampled.csv', index=False)
    if not os.path.exists(os.path.join(test_dir, 'y_val.csv')):
        pd.DataFrame(y_val).to_csv('y_val.csv', index=False)
    if not os.path.exists(os.path.join(test_dir, 'y_test.csv')):
        pd.DataFrame(y_test).to_csv('y_test.csv', index=False)

# The preprocessing pipeline 
def preprocess_and_save(file_path):
    df = load_and_inspect_data(file_path)
    df, label_encoder = encode_categorical_column(df)
    df_no_outliers = cap_outliers(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_no_outliers)
    X_train_resampled, y_train_resampled = handle_imbalance(X_train, y_train)
    X_train_resampled_scaled, X_val_scaled, X_test_scaled = scale_features(X_train_resampled, X_val, X_test)
    save_data_to_csv(X_train_resampled_scaled, X_val_scaled, X_test_scaled, y_train_resampled, y_val, y_test)
    return X_train_resampled_scaled, X_val_scaled, X_test_scaled, y_train_resampled, y_val, y_test, label_encoder

if __name__ == "__main__":
    file_path = 'maternal_health_risk.csv'
    X_train_resampled_scaled, X_val_scaled, X_test_scaled, y_train_resampled, y_val, y_test, label_encoder = preprocess_and_save(file_path)
    print("\nPreprocessing Completed!")
    print(f"Shapes:\nX_train: {X_train_resampled_scaled.shape}\nX_val: {X_val_scaled.shape}\nX_test: {X_test_scaled.shape}")
    print(f"y_train: {y_train_resampled.shape}\ny_val: {y_val.shape}\ny_test: {y_test.shape}")