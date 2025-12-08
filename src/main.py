# ------------------ IMPORT LIBRARIES --------------------------
import os
import sys
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException, status, Depends
from typing import Annotated

# --- FIX: CORRECTED IMPORTS for the new flattened 'src' structure ---
# We no longer need src.api.schemas or src.core.prediction.
# We import directly from the sibling files within the src/ package.
from schemas import LoanPredictionInput, LoanPredictionOutput
from prediction import load_pipeline, predict_default, MODEL_VERSION, pipeline


# ------------ Configuration Pin -------------------------------
SECRET_API_KEY = "SUPER_SECRET_KEY_12345"


# ------------------- FastAPI App ------------------------------
app = FastAPI(
    title="Loan Default Prediction API",
    version=MODEL_VERSION,
    description="A secure service for predicting loan default risk using a Random Forest Pipeline."
)


# ---------- DEPENDENCY INJECTION ------------------------------
def verify_api_key(x_api_key: str = Header(None)):
    """
    Checks for a valid API key in the X-API-key header.
    """

    if x_api_key is None or x_api_key != SECRET_API_KEY:
        raise HTTPException (
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key in X-API-Key header"
        )
    return True

AuthDep = Annotated[bool, Depends(verify_api_key)]


# ------------ APPLICATION LIFECYCLE EVENTS -------------------
@app.on_event("startup")
async def startup_event():
    """Run model loading when the application starts."""
    print("⏳ Running application startup routines...")
    # FIX: Must CALL the function using parentheses!
    load_pipeline() 
    print("✅ Startup complete. API is ready to serve traffic.")


# -------------- ENDPOINTS -----------------------------------

# health check route
@app.get("/", tags=["Health Check"])
def health_check():
    """Simple health check to verify API status and model version."""
    status_detail = "Ready" if pipeline is not None else "Model Not Loaded"

    return {
        "status": status_detail,
        "api_version": MODEL_VERSION,
        "model_loaded": pipeline is not None
    }


@app.post(
    "/predict", 
    response_model=LoanPredictionOutput, 
    tags=["Prediction"],
    dependencies=[Depends(verify_api_key)]
)
def predict(
    input_data: LoanPredictionInput
):
    """
    Accepts raw loan features and returns the prediction result and risk assessment.
    """
    try:
        # Call the core prediction logic
        result = predict_default(input_data)
        return result
    except RuntimeError as e:
        # Catch the specific error raised if the model failed to load
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail=str(e)
        )
    except Exception as e:
        # Catch any unexpected errors during prediction
        print(f"Prediction Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="An unexpected error occurred during prediction processing."
        )