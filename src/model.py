import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
)

from src.preprocessing import preprocess_and_save

def train_and_evaluate_model(filepath, model_dir='./models'):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")
        
    (X_train_resampled_scaled, X_val_scaled, X_test_scaled, y_train_resampled, y_val, y_test, label_encoder) = preprocess_and_save(filepath)
    
    # Initializing the Random forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_resampled_scaled, y_train_resampled)
    
    #Predictions on val set
    y_val_pred = model.predict(X_val_scaled)
    
    # Evaluation on val set
    mse = mean_squared_error(y_val, y_val_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_confusion_matrix = confusion_matrix(y_val, y_val_pred)
    val_classification_report = classification_report(
        y_val, y_val_pred, target_names=["High Risk", "Low Risk", "Mid Risk"]
    )

    print("\nValidation Results:")
    print("Mean Squared Error (MSE):", mse)
    print("Validation Accuracy:", val_accuracy)
    print("Validation Confusion Matrix:\n", val_confusion_matrix)
    print("Validation Classification Report:\n", val_classification_report)

    # Predictions on test set
    y_test_pred = model.predict(X_test_scaled)

    # Evaluate on test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
    test_classification_report = classification_report(
        y_test, y_test_pred, target_names=["High Risk", "Low Risk", "Mid Risk"]
    )

    print("\nTest Results:")
    print("Test Accuracy:", test_accuracy)
    print("Test Confusion Matrix:\n", test_confusion_matrix)
    print("Test Classification Report:\n", test_classification_report)

    # Save the trained model as a pickle file
    model_path = os.path.join(model_dir, "random_forest_model.pkl")
    if not os.path.exists(model_path):
        with open(model_path, "wb") as model_file:
            pickle.dump(model, model_file)
        print(f"\nModel saved to: {model_path}")
    else:
        print(f"\nModel already exists at: {model_path}. Skipping save.")

    print("Model training and evaluation completed!")
    return model, label_encoder



if __name__ == "__main__":
    file_path = "maternal_health_risk.csv"  
    trained_model, label_encoder = train_and_evaluate_model(file_path)