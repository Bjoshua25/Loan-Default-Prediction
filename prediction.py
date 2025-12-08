# ----------------- IMPORT LIBRARIES ---------------------------
import os
import sys
import joblib
import pandas as pd
from pathlib import Path
from schemas import LoanPredictionInput, LoanPredictionOutput


# ------------- DIRECTORY HANDLING ----------------------------
# parent root folder
parent_dir = Path(__file__).resolve().parent

# save path to system
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))




# ----------- GLOBAL MODEL VARIABLES ---------------------------
MODEL_PATH = parent_dir / "models" / "best_rf_pipeline.pkl"
MODEL_VERSION = "1.0.0"

# loaded model (Global)
pipeline = None




# -------------- LOAD SAVED MODEL -----------------------------
def load_pipeline():
    """
    About: 
        loads the full scikit-learn peipeline(preprocessor + model) from disk.
    """

    global pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
        print(f"Pipeline (Version {MODEL_VERSION}) loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_PATH}. Check file Path.")
        pipeline = None
    except Exception as e:
        print(f"ERROR loading model: {e}")
        pipeline = None





# ----------- PREDICTION FUNCTION -----------------------------
def predict_default(input_data: LoanPredictionInput) -> LoanPredictionOutput:
    """
    About:
        Performs the prediction using the loaded model pipeline.
    Input:
        input_data (LoanPredictionInput) - The pydantic model containing the raw features.
    output:
        LoanPredictionOutput: The structured prediction result.
    """

    if pipeline is None:
        raise RuntimeError ("Model Pipeline is Not loaded. Cannot Perform Prediction.")
    
    # Convert Pydantic object to DataFrame
    input_dict = input_data.model_dump()
    input_df = pd.DataFrame([input_dict])

    # Make probability prediction
    probability = pipeline.predict_proba(input_df)[:, 1][0]

    # Get Binary Prediction
    prediction = pipeline.predict(input_df)[0]

    # Obtin  human-readable result
    is_default = int(prediction)
    recommendation = (
        "High Risk: Loan Default Predicted" if is_default == 1
        else "Low Risk: Loan Repaymnet Predicted"
    )

    return LoanPredictionOutput(
        is_default = is_default,
        default_probability = float(probability),
        recommendation = recommendation,
        model_version = MODEL_VERSION
    )