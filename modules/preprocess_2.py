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





# ----------------------- IMPORT FUNCTIONS FROM SCRIPT ---------------------
from preprocess import load_and_split_data
from preprocess import transform_data
from preprocess import get_transformed_df




# ------------------ UPDATED DATA CLEANING FUNCTION ------------------------
def clean_data(df):

    return df




# ----------------------- DEFINE FEATURE LISTS -----------------------
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





# -------------- UPDATED PREPROCESSING PIPELINE -----------------

def get_preprocessing_pipeline():

    # Log pipeline

    # Numerical pipeline

    # Categorical pipeline

    # Combine All Pipeline using ColumnTransformer

    # Full Transformation Pipeline

    return full_pipeline




# -------------------------- TRANSFORM DATASET -----------------------------

def transform_data():

    """
    About: 
        A function that load, split, apply transformation pipeline on dataset
    Input:
        No Args
    Output:
        X_train_trans (numpy.ndarray): transformed X_train 
        X_test_trans (numpy.ndarray): transformed X_test
        pipeline (Pipeline): the preprocessing pipeline used for transformation
    """

    # Load and Split data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(filepath = dataset_dir)

    # build pipeline
    pipeline = get_preprocessing_pipeline()

    # fit and transform train set
    X_train_trans = pipeline.fit_transform(X_train)

    # transform the test set
    X_test_trans = pipeline.transform(X_test)

    return X_train_trans, X_test_trans, pipeline





# -------------------- PROPROCESSED DATAFRAME ------------------------

def get_transformed_df():

    # load and split Dataset
    X_train, X_test, y_train, y_test= load_and_split_data(filepath= dataset_dir)

    # transform feature matrices
    X_train_trans, X_test_trans, pipeline= transform_data()

    # Access OneHot Encoder pipeline
    onehot_transformer = pipeline["preprocessor"].named_transformers_['cat']['onehot']

    # Extract feature names from OneHot Transformer
    cat_cols = list(onehot_transformer.get_feature_names_out(categorical_cols))

    # Compilation of all Processed feature names
    processed_feat = (
        log_transform_cols +
        numerical_cols +
        cat_cols
    )

    # dataframe of the processed data
    df_processed = pd.DataFrame(
        X_train_trans,
        columns= processed_feat,
        index = X_train.index
    )

    # Add Target feature back to the processed dataframe
    df_processed = pd.concat([df_processed, y_train], axis=1)

    return df_processed