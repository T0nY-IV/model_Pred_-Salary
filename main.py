Python 3.13.9 (tags/v3.13.9:8183fa5, Oct 14 2025, 14:09:13) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
>>> from fastapi import FastAPI
... import joblib
... import pandas as pd
... 
... app = FastAPI()
... 
... model = joblib.load("gradianBoostingRegression.pkl")
... 
... @app.post("/predict")
... def predict(data: dict):
...     df = pd.DataFrame([data])
...     pred = model.predict(df)[0]
