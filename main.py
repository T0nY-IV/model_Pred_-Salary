from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = FastAPI()

# Load model
try:
    model = joblib.load("gradianBoostingRegression.pkl")
except FileNotFoundError:
    raise FileNotFoundError("Model file 'gradianBoostingRegression.pkl' not found")

# Load training data to fit encoders and scaler
# This should match the data used for training
try:
    df_train = pd.read_csv("HRDataset_same_structure.csv")
    df_train = df_train.drop_duplicates()

    # Drop unnecessary columns (same as in notebook)
    df_train = df_train.drop(columns=[
        'Employee_Name', 'EmpID', 'MarriedID', 'MaritalStatusID', 'GenderID',
        'EmpStatusID', 'DeptID', 'PerfScoreID', 'PositionID', 'ManagerID',
        'Zip', 'DOB', 'DateofTermination', 'TermReason', 'LastPerformanceReview_Date'
    ], errors='ignore')

    df_train.dropna(inplace=True)
    df_train['Salary'] = df_train['Salary'] / 12

    # Prepare features (same as notebook)
    X_train = df_train.drop('Salary', axis=1)
    Y_train = df_train['Salary']

    # Fit label encoders for categorical features
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        label_encoders[col] = le

    # Convert to numpy array and fit scaler
    X_train_array = X_train.values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_array)

    # Define feature order (important for prediction)
    feature_order = X_train.columns.tolist()
    
except FileNotFoundError:
    raise FileNotFoundError("Training data file 'HRDataset_same_structure.csv' not found. Please ensure it's available for fitting encoders and scaler.")
except Exception as e:
    raise RuntimeError(f"Error initializing preprocessing: {str(e)}")

@app.post("/predict")
def predict(data: dict):
    try:
        # Create DataFrame from input
        df = pd.DataFrame([data])
        
        # Ensure all required columns are present
        missing_cols = [col for col in feature_order if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Reorder columns to match training data
        df = df[feature_order]
        
        # Encode categorical features
        for col in categorical_cols:
            if col in df.columns:
                le = label_encoders[col]
                # Handle unseen categories
                try:
                    df[col] = le.transform(df[col].astype(str))
                except ValueError as e:
                    # If category not seen during training, assign a default value
                    # Use the first label (0) as default
                    df[col] = 0
        
        # Convert to numpy array and scale
        X_array = df.values.astype(float)
        X_scaled = scaler.transform(X_array)
        
        # Make prediction
        pred = model.predict(X_scaled)[0]
        return {"prediction": float(pred)}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )
