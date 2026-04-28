import pandas as pd
import joblib
from pathlib import Path

from preprocess import preprocess_data


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "loan_model.pkl"


def predict_loan():
    model = joblib.load(MODEL_PATH)

    sample_data = pd.DataFrame([{
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 1000,
        "LoanAmount": 130,
        "Credit_History": 1,
        "Education": "Graduate",
        "Self_Employed": "No"
    }])

    sample_data = preprocess_data(sample_data)
    prediction = model.predict(sample_data)[0]

    if prediction == 1:
        print("Loan Prediction: Approved")
    else:
        print("Loan Prediction: Rejected")


if __name__ == "__main__":
    predict_loan()
