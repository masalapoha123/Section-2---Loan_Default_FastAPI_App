from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.prediction import predict
from typing import Optional
from pydantic import BaseModel, Field, validator,ConfigDict


# Initialize FastAPI
app = FastAPI(
    title="Loan Default Prediction API",
    description="API to predict whether a loan applicant will default or not",
    version="1.0"
)

# Input schema (UPDATE feature names as per your dataset exactly)
class LoanInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",          # exact schema: reject unknown fields
        validate_by_name=True,   # allow population by field name
        populate_by_name=True    # allow alias + field-name inputs
    )

    ID: int
    year: int
    loan_limit: str
    Gender: str
    approv_in_adv: str
    loan_type: str
    loan_purpose: str
    Credit_Worthiness: str
    open_credit: str
    business_or_commercial: str
    loan_amount: float

    # These were np.nan in your sample -> in API JSON send null (None)
    rate_of_interest: Optional[float] = None
    Interest_rate_spread: Optional[float] = None
    Upfront_charges: Optional[float] = None

    term: float
    Neg_ammortization: str
    interest_only: str
    lump_sum_payment: str
    property_value: float
    construction_type: str
    occupancy_type: str
    Secured_by: str
    total_units: str
    income: float
    credit_type: str
    Credit_Score: float

    # exact JSON key has a hyphen
    co_applicant_credit_type: str = Field(..., alias="co-applicant_credit_type")

    age: str
    submission_of_application: str
    LTV: float
    Region: str
    Security_Type: str
    dtir1: float

@app.get("/")
def home():
    return {
        "status": "success",
        "message": "Loan Default Prediction API is running"
    }

# Prediction route
@app.post("/predict")
def predict_loan_default(input_data: LoanInput):
    try:
        # ✅ Pydantic v2: convert to dict (use alias key: "co-applicant_credit_type")
        data = input_data.model_dump(by_alias=True)

        # Call prediction function
        prediction = predict(data)

        # ✅ Make response JSON-safe (handles numpy types)
        pred_val = prediction.get("pred", prediction.get("prediction", prediction))
        prob_val = prediction.get("prob", prediction.get("probability", None))

        if pred_val is not None:
            try: pred_val = int(pred_val)
            except: pass

        if prob_val is not None:
            try: prob_val = float(prob_val)
            except: pass

        return {
            "status": "success",
            "prediction": pred_val,
            "probability": prob_val,
            "message": "Prediction generated successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))