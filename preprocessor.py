# preprocessor.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pickle

def preprocess(df):
    """
    Preprocess the customer churn dataset.
    
    Parameters:
    df (DataFrame): Raw customer churn data
    
    Returns:
    tuple: encoders, X_train_smote, y_train_smote, X_test, y_test
    """
    df = df.copy()

    # Drop customerID if present
    if "customerID" in df.columns:
        df.drop(columns="customerID", inplace=True)

    # Clean TotalCharges and convert to float
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0", "": "0.0"})
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce').fillna(0.0)

    # Encode target variable
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        # Handle any remaining non-numeric values
        df["Churn"] = df["Churn"].fillna(0).astype(int)

    # Encode categorical columns
    object_columns = df.select_dtypes(include="object").columns
    encoders = {}
    for column in object_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        encoders[column] = le

    # Save encoders for later use
    with open("encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    # Split features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    return encoders, X_train_smote, y_train_smote, X_test, y_test

