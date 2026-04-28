import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from preprocess import preprocess_data


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "loan_data.csv"
MODEL_PATH = BASE_DIR / "models" / "loan_model.pkl"


def train_model():
    df = pd.read_csv(DATA_PATH)
    df = preprocess_data(df)

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model trained successfully.")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
