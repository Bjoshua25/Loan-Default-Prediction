# ------------------ IMPORT LIBRARIES --------------------------
import sys
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException, status, Depends
from typing import Annotated

# Handle directory
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# import custom modules
from src.schemas import LoanPredictionInput, LoanPredictionOutput
from src.prediction import load_pipeline, predict_default, MODEL_VERSION, pipeline




# ------------ Configuration Pin -------------------------------
SECRET_API_KEY = "SUPER_SECRET_KEY_12345"


# ------------------- FastAPI App ------------------------------
app = FastAPI(
    title = "Loan Default Prediction API",
    version= MODEL_VERSION,
    description= "A secure service for predicting loan default risk using a Random Forest Pipeline."
)



# ---------- DEPENDENCY INJECTION ------------------------------
def verify_api_key(x_api_key: str = Header(None)):
    """
    Checks for a valid API key in the X-API-key header.
    """

    if x_api_key is None or x_api_key != SECRET_API_KEY:
        raise HTTPException (
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail= "Invalid or missing API key in X_API_key header"
        )
    return True

AuthDep = Annotated[bool, Depends(verify_api_key)]




# ------------ APPLICATION LIFECYCLE EVENTS -------------------
@app.on_event("startup")
async def startup_event():
    """Run model loading when the application starts."""
    print("Running application startup routines...")
    load_pipeline
    print("Startup complete. API is ready to serve traffic.")



# -------------- ENDPOINTS -----------------------------------

# health check route
@app.get("/", tags= ["Health Check"])
def health_check():
    status_detail = "Ready" if pipeline is not None else "Model Not Loaded"

    return {
        "status": status_detail,
        "api_version": MODEL_VERSION,
        "model_loaded": pipeline is not None
    }





# ------------------ Simple Endpoint ---------------------------
@app.get("/")
def home_endpoint():

    return "Welcome to this Page......."