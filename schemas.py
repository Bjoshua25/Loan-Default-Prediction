# ----------------- IMPORT LIBRARIES ---------------------------
from pydantic import BaseModel
from typing import Optional 



# ------------------- INPUT SCHEMA -----------------------------
class LoanPredictionInput(BaseModel):

    # features that are handled/dropped by the pipeline
    ID:                         Optional[int]= None
    year:                       Optional[int] = None
    business_or_commercial:     Optional[str] = None
    property_value:             Optional[float] = None
    rate_of_interest:           Optional[float] = None
    age:                        Optional[str] = None
    Credit_Score:               Optional[float] = None

    # Features that undergo Log/Scaling (Numerical)
    loan_amount:            int
    income:                 float
    Upfront_charges:        float
    Interest_rate_spread:   Optional[float] = None
    dtir1:                  Optional[float]= None
    LTV:                    Optional[float] = None
    term:                   int

    # features that undergo One-Hot Encoding (Categorical)
    loan_limit:         Optional[str] = None
    Gender:             Optional[str]= None
    approv_in_adv:      Optional[str] = None
    loan_type:          Optional[str] = None
    loan_purpose:       Optional[str] = None
    Credit_Worthiness:  Optional[str] = None
    open_credit:        Optional[str] = None
    Neg_ammortization:  Optional[str] = None
    interest_only:      Optional[str] = None
    lump_sum_payment:   Optional[str] = None
    construction_type:  Optional[str] = None
    occupancy_type:     Optional[str] = None
    Secured_by:         Optional[str] = None
    total_units:        Optional[str] = None
    credit_type:        Optional[str] = None
    co_applicant_credit_type:   Optional[str] = None
    submission_of_application:  Optional[str] = None
    Region:             Optional[str] = None
    Security_Type:      Optional[str] = None




# ----------------- OUTPUT SCHEMA ----------------------
class LoanPredictionOutput(BaseModel):
    is_default: int
    default_probability: float
    recommendation: str
    model_version: str
