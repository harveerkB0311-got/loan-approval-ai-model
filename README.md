# Loan Approval AI Model

This project predicts loan approval decisions using machine learning. It demonstrates data cleaning, feature engineering, model training, evaluation, and API deployment using FastAPI.

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Joblib

## Features
- Data preprocessing
- Machine learning model training
- Loan approval prediction
- REST API endpoint
- Clean GitHub-ready folder structure

## Project Structure
```text
loan-approval-ai-model/
├── api/
│   └── app.py
├── data/
│   └── loan_data.csv
├── models/
│   └── loan_model.pkl
├── notebooks/
├── src/
│   ├── train_model.py
│   ├── predict.py
│   └── preprocess.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

### 1. Install packages
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python src/train_model.py
```

### 3. Test prediction
```bash
python src/predict.py
```

### 4. Run API
```bash
uvicorn api.app:app --reload
```

Then open:
```text
http://127.0.0.1:8000/docs
```
