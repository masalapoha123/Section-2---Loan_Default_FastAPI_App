import joblib
import numpy as np
import pandas as pd


# Extract safely
bundle = joblib.load(r"D:\OneDrive - Coforge Limited\Documents\CAPSTONE PROJECT 1\Section 2 - Loan_Default_FastAPI_App\model\loan_default_bundle.pkl")
model = bundle["model"]
raw_cols = bundle["features_raw"]

def predict(payload: dict):
        # ✅ Create DataFrame
    X_in = pd.DataFrame([payload])

    # ensure same raw schema
    for c in raw_cols:
        if c not in X_in.columns:
            X_in[c] = np.nan
    X_in = X_in[raw_cols]

    pred = model.predict(X_in)[0]
    prob = model.predict_proba(X_in)[0, 1] if hasattr(model, "predict_proba") else None
    print(pred, prob)
    return {
        'pred':pred,
        'prob':prob
    }



# ✅ Test
if __name__ == "__main__":
    sample_input = {
    "ID": 24890,
    "year": 2019,
    "loan_limit": "cf",
    "Gender": "Male",
    "approv_in_adv": "nopre",
    "loan_type": "type1",
    "loan_purpose": "p1",
    "Credit_Worthiness": "l1",
    "open_credit": "nopc",
    "business_or_commercial": "nob/c",
    "loan_amount": 116500,
    "rate_of_interest": np.nan,
    "Interest_rate_spread": np.nan,
    "Upfront_charges": np.nan,
    "term": 360.0,
    "Neg_ammortization": "not_neg",
    "interest_only": "not_int",
    "lump_sum_payment": "not_lpsm",
    "property_value": 118000.0,
    "construction_type": "sb",
    "occupancy_type": "pr",
    "Secured_by": "home",
    "total_units": "1U",
    "income": 1740.0,
    "credit_type": "EXP",
    "Credit_Score": 758,
    "co-applicant_credit_type": "CIB",
    "age": "25-34",
    "submission_of_application": "to_inst",
    "LTV": 98.72881356,
    "Region": "south",
    "Security_Type": "direct",
    "dtir1": 45.0
}
