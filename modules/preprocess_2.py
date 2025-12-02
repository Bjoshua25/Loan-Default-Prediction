# ---------------------- IMPORT LIBRARIES --------------------------
import os 
import sys 
import warnings
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




# Handle warnings
warnings.filterwarnings('ignore')




# --------------------- HANDLING DIRECTORY -------------------------

# parent folder director
parent_dir = Path(__file__).parent.parent

# data directory
data_dir = parent_dir / "data"

# dataset directory
dataset_dir = data_dir / "Loan_Default.csv"

# Add parent directory to system
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))





# ----------------------- DEFINE FEATURE LISTS -----------------------
# Log Transform Columns
log_transform_cols = [
    'loan_amount',
    'income',
    'Upfront_charges',
    'Interest_rate_spread',
    'dtir1'
]

# Numerical columns for imputation and scaling
numerical_cols = [
    'LTV',
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
    'Security_Type',
    'term', 
    'Credit_Score_Group' 
]




# ----------------- LOAD AND SPLIT DATA --------------------------
def load_and_split_data(filepath= dataset_dir):

    """
    About: 
        Function to load DataFrame from filepath, then splits dataset into train and test sets using 80/20 split rule.
    Input:
        filepath (str): Absolute file path
    Output:
        X_train (pd.DataFrame) 
        X_test (pd.DataFrame) 
        y_train (pd.Series)
        y_test (pd.Series)
    """

    # create dataframe 
    df = pd.read_csv(filepath)

    # Prediction Matrix
    X = df.drop(columns= "Status")

    # Target vector
    y = df["Status"]

    # train_test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    return X_train, X_test, y_train, y_test




# ------------------ UPDATED DATA CLEANING FUNCTION ------------------------
def clean_data(df):
    """
    About:
        Function to clean the data and perform feature engineering.
    Input:
        DataFrame: 
    Output:
        Cleaned DataFrame
    """

    # ------- DROPPING NON-PREDICTIVE COLUMNS -------
    # create a copy of the dataframe
    df = df.copy()

    # drop non-predictive columns
    df = df.drop(columns = ["ID", "year"], errors='ignore')

    # Drop collinear numerical features based on EDA
    df = df.drop(columns = ["property_value", "rate_of_interest"], errors='ignore')

    # Drop redundant categorical feature based on EDA
    df = df.drop(columns = ["business_or_commercial"], errors='ignore')


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
    df = df.drop(columns=['age'], errors='ignore')

    # Outlier capping for LTV (Loan-to-Value)
    LTV_CAP = 150.0
    df['LTV'] = np.where(df['LTV'] > LTV_CAP, LTV_CAP, df['LTV'])


    # ---------- CREDIT SCORE BINNING -------
    # Convert the continuous Credit_Score to ordinal categories
    bins = [0, 580, 670, 740, 800, 1000] 
    labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']

    # create categorical feature and drop the original numerical score
    df["Credit_Score_Group"] = pd.cut(
        df["Credit_Score"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    ).astype('object')
    df = df.drop(columns=['Credit_Score'], errors='ignore') 

    # Convert TERM to categorical variable to handle extreme skewness
    df['term'] = df['term'].astype(object)

    return df





# -------------- UPDATED PREPROCESSING PIPELINE -----------------

def get_preprocessing_pipeline():
    """
    About:
        Function to create preprocessing pipeline for numerical and categorical features.
    Input:
        DataFrame
    Output:
        Preprocessing Pipeline
    """

    # Log pipeline
    log_pipeline = Pipeline(
        steps= [
            ('imputer', SimpleImputer(strategy='median')), 
            ('log_transform', FunctionTransformer(np.log1p, validate=False)), 
            ('scalar', StandardScaler())
        ]
    )

    # Numerical pipeline
    num_pipeline = Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy = 'median')),
            ('scalar', StandardScaler())
        ]
    )

    # Categorical pipeline
    cat_pipeline = Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
        ]
    )

    # Combine All Pipeline using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('log_num', log_pipeline, log_transform_cols),
            ('num', num_pipeline, numerical_cols),
            ('cat', cat_pipeline, categorical_cols),
        ],
        remainder= 'drop'
    )

    # Full Transformation Pipeline
    full_pipeline = Pipeline(
        steps = [
            ('data_cleaning', FunctionTransformer(clean_data, validate=False)),
            ('preprocessor', preprocessor),
            ('final_imputer', SimpleImputer(strategy='median'))
        ]
    )

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
    X_train, X_test, y_train, y_test = load_and_split_data(filepath=dataset_dir)

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















if __name__ == "__main__":
    processed_df = get_transformed_df()

    print("\n=======================================================")
    print("Preprocessed Training DataFrame Successfully Created!")
    print("=======================================================\n")

    # Display dataframe shape
    print(f"Shape of Processed DataFrame (Training Set): {processed_df.shape}")

    # Display first 5 rows of the dataframe
    print("\nHead of the Processed DataFrame:")
    print(processed_df.head())


    # list of first 10 column names
    print("\nFirst 15 feature names")
    print(processed_df.columns[:15])