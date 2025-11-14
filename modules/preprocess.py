# ---------------------- IMPORT LIBRARIES --------------------------
import os 
import sys 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer






# --------------------- HANDLING DIRECTORY -------------------------

# parent folder director
parent_dir = Path.cwd().parent

# data directory
data_dir = parent_dir / "data"

# dataset directory
dataset_dir = data_dir / "Loan_Default.csv"

# Add parent directory to system
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))








# ------------------ DATA CLEANING FUNCTION ------------------------
def clean_data(df):

    """
    About:
        Function to clean the data and perform feature engineering.
    Input:
        DataFrame: 
    Output:
        Cleaned DataFrame
    """
    
    # create a copy of the dataframe
    df = df.copy()

    # drop non-predictive columns
    df = df.drop(columns = ["ID", "year"])

    # ---- FUNCTION CONVERTING AGE RANGE TO MIDPOINT ----
    def age_midpoint (age_range):
        if pd.isna(age_range):
            return np.nan
        elif age_range == '<25':
            return 20.0
        elif age_range == '>74':
            return 75.0
        elif '-' in age_range:
            lower, upper = map(int, age_range.split('-'))
            return (lower + upper) / 2
        return np.nan
    
    # Apply age conversion and drop the original 'age' column
    df['age_numerical'] = df['age'].apply(age_midpoint)
    df = df.drop(columns=['age'])

    # Outlier capping for LTV (Loan-to-Value)
    LTV_CAP = 150.0

    # Use fillna(df['LTV']) to handle potential NaNs before capping
    df['LTV'] = np.where(df['LTV'] > LTV_CAP, LTV_CAP, df['LTV'])

    return df






# ------------------ PREPROCESSING PIPELINE ------------------------

def preprocessing_pipeline(df):
    """
    About:
        Function to create preprocessing pipeline for numerical and categorical features.
    Input:
        DataFrame:
    Output:
        Preprocessing Pipeline
    """

    # ------- DEFINE FEATURE LISTS -------
    # Log Transform Columns
    log_transform_cols = [
        'loan_amount',
        'property_value',
        'income',
        'Upfront_charges'
    ]

    # Numerical columns for imputation and scaling
    numerical_cols = [
        'rate_of_interest',
        'Interest_rate_spread',
        'term',
        'Credit_Score',
        'LTV',
        'dtir1',
        'age_numerical'
    ]


    # Categorical columns for imputation and One-Hot Encoding
    categorical_cols = [
        'loan_limit',
        'Gender',
        'approv_in_adv',
        'loan_type',
        'loan_purpose',
        'Credit_Worthiness',
        'open_credit',
        'business_or_commercial',
        'Neg_ammortization',
        'interest_only',
        'lump_sum_payment',
        'construction_type',
        'occupancy_type',
        'Secured_by',
        'total_units',
        'credit_type',
        'co-applicant_credit_type',
        'submission_of_application',
        'Region',
        'Security_Type'
    ]

    # ------- LOG TRANSFROMATION -----
    log_pipeline = Pipeline(
        steps= [
            ('imputer', SimpleImputer(strategy='median')),
            ('log_transform', FunctionTransformer(np.log1p, validate=False)),
            ('scalar', StandardScaler())
        ]
    )

    # ------ NUMERICAL PIPELINE ------
    num_pipeline = Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy = 'median')),
            ('scalar', StandardScaler())
        ]
    )

    # ------ CATEGORICAL PIPELINE ------
    cat_pipeline =  Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]
    )

    # ------ COMBINE ALL PIPELINES USING COLUMN TRANSFORMER ------
    preprocessor = ColumnTransformer(
        transformers=[
            ('log_num', log_pipeline, log_transform_cols),
            ('num', num_pipeline, numerical_cols),
            ('cat', cat_pipeline, categorical_cols)
        ],
        remainder= "drop"
    )

    # ------ COMBINE ALL PIPELINES USING COLUMN TRANSFORMER ------
    preprocessor = ColumnTransformer(
        transformers=[
            ('log_num', log_pipeline, log_transform_cols),
            ('num', num_pipeline, numerical_cols),
            ('cat', cat_pipeline, categorical_cols)
        ],
        remainder= "drop"
    )

    # --------- FULL TRANSFORMATION PIPELINE---------
    full_pipeline = Pipeline(
        steps = [
            ('data_cleaning', FunctionTransformer(clean_data, validate=False)),
            ('preprocessor', preprocessor)
        ]
    )

    return full_pipeline


