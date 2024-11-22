from fastapi import FastAPI, HTTPException, status, UploadFile, File, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os

app = FastAPI()

# Enable CORS for testing with Postman or a frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pipeline components
model = joblib.load('./models/random_forest_model.pkl')
scaler = joblib.load('./data/scaler/scaler.pkl')
label_encoder = joblib.load('./data/scaler/label_encoder.pkl')

class PredictionData(BaseModel):
    Age: int = Field(gt=11, lt=71, description="Age must be between 12 and 70.")
    SystolicBP: int = Field(ge=60, le=200, description="SystolicBP must be between 60 and 200.")
    DiastolicBP: int = Field(ge=40, le=140, description="DiastolicBP must be between 40 and 140.")
    BS: float = Field(ge=2.0, le=15.0, description="Blood Sugar (BS) must be between 2.0 and 15.0.")
    BodyTemp: float = Field(ge=90.0, le=110.0, description="Body temperature must be between 90.0 and 110.0.")
    HeartRate: int = Field(ge=50, le=120, description="Heart rate must be between 50 and 120.")
    

@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    return {"message": "Welcome to the Maternal Health API!"}

@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict(data: PredictionData):
    """
    Predict maternal health risk level based on input features.
    """
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Preprocess data
        input_scaled = scaler.transform(input_df)

        # Predict
        predictions = model.predict(input_scaled)
        predictions = predictions.astype(int)
        decoded_predictions = label_encoder.inverse_transform(predictions)

        return {"Your Predicted Risk Level is: ": decoded_predictions[0]}
    
    except Exception as err:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(err))
    
upload_dir = './data/uploads'
os.makedirs(upload_dir, exist_ok=True)

@app.post("/upload-data", status_code=status.HTTP_200_OK)
async def upload_data(file: UploadFile = File(...)):
    """
    Upload new data for retraining the model
    
    Args:
    file (UploadFile): The uploaded file (CSV or Excel).
    
    Returns:
    JSONResponse: A success message or an error message.
    """
    try:
        # Check the file's content type
        allowed_types = ["text/csv", "application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Invalid file type. Only CSV or Excel files are allowed.")
        
        file_location = os.path.join(upload_dir, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())
            
        normalized_file_path = os.path.normpath(file_location)

        # Read the file into a Pandas DataFrame
        if file.content_type == "text/csv":
            df = pd.read_csv(normalized_file_path)
        else:
            df = pd.read_excel(normalized_file_path)

        # Validate the columns (assuming specific columns are expected)
        expected_columns = {"Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "RiskLevel"}
        if not expected_columns.issubset(df.columns):
            os.remove(normalized_file_path)
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns. Expected columns are: {expected_columns}",
            )

        return {
            "message": f"File {file.filename} uploaded successfully!",
            "file_path": normalized_file_path.replace("\\", "/"),
        }
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded file is empty or invalid.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/retrain", status_code=status.HTTP_200_OK)
async def retrain(file_path: str = Body(...)):
    """
    Retrain the model using uploaded data and evaluate it.
    
    Args:
    - file_path (str): Path to the uploaded data file.
    
    Returns:
    - dict: Confirmation of retraining, evaluation metrics, and the new model version.
    """
    from src.model import train_and_evaluate_model
    import pickle
    
     
    try:
        file_path = os.path.normpath(file_path)
        # Ensure the file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found.")
        
        
        # Reuse the existing training pipeline (includes preprocessing)
        model, label_encoder = train_and_evaluate_model(file_path)

        # Versioning: Save the retrained model with a new version number
        version = 1
        model_dir = "./models"
        os.makedirs(model_dir, exist_ok=True)
        while os.path.exists(f"{model_dir}/random_forest_model_v{version}.pkl"):
            version += 1

        model_filename = f"{model_dir}/random_forest_model_v{version}.pkl"
        with open(model_filename, "wb") as model_file:
            pickle.dump(model, model_file)

        return {
            "message": "Model retrained successfully!",
            "model_version": version,
            "model_path": model_filename,
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(err)}")
