import numpy as np
import pandas as pd

NUM_FEATURES = ["ApplicantIncome_log", "CoapplicantIncome_log", "LoanAmount_log",
                "Total_Income_log", "Loan_Amount_Term", "Income_to_Loan", "EMI_to_Income"]
CAT_FEATURES = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
BIN_FEATURES = ["Credit_History"]


def add_features(df):
    df = df.copy()
    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["EMI"] = df["LoanAmount"] * 1000 / df["Loan_Amount_Term"]
    df["Income_to_Loan"] = df["Total_Income"] / df["LoanAmount"]
    df["EMI_to_Income"] = df["EMI"] / df["Total_Income"]
    for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Total_Income"]:
        df[col + "_log"] = np.log1p(df[col])
    return df