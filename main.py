from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("gradianBoostingRegression.pkl")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return {"prediction": float(pred)}