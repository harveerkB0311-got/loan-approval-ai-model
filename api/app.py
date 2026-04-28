import sys
from pathlib import Path

import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from preprocess import preprocess_data


MODEL_PATH = BASE_DIR / "models" / "loan_model.pkl"

app = FastAPI(title="Loan Approval AI API")


class LoanApplication(BaseModel):
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Credit_History: int
    Education: str
    Self_Employed: str


@app.get("/")
def home():
    return {"message": "Loan Approval AI API is running"}


@app.post("/predict")
def predict(application: LoanApplication):
    model = joblib.load(MODEL_PATH)

    input_data = pd.DataFrame([application.dict()])
    input_data = preprocess_data(input_data)

    prediction = model.predict(input_data)[0]

    result = "Approved" if prediction == 1 else "Rejected"

    return {
        "loan_prediction": result
    }
