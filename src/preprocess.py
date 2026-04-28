import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns into numeric format."""

    df = df.copy()

    df["Education"] = df["Education"].map({
        "Graduate": 1,
        "Not Graduate": 0
    })

    df["Self_Employed"] = df["Self_Employed"].map({
        "Yes": 1,
        "No": 0
    })

    if "Loan_Status" in df.columns:
        df["Loan_Status"] = df["Loan_Status"].map({
            "Y": 1,
            "N": 0
        })

    return df
