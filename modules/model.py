# ---------------------- IMPORT LIBRARIES --------------------------
import os 
import sys 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import joblib
import fastapi
import uvicorn

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report,accuracy_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')



# --------------------- HANDLING DIRECTORY -------------------------

try:
    # Use Path(__file__) if the script is run directly
    current_script_dir = Path(__file__).resolve().parent
    parent_dir = current_script_dir.parent.parent # Assuming model.py is in modules/
    
    # If the script is run from an interactive environment where __file__ is unavailable,
    # use a fallback, though the previous directory logic often fails in complex setups.
    # We will rely on the structure: .../Loan Default Prediction/modules/model.py
except NameError:
    parent_dir = Path.cwd().parent.parent

# Set specific directories
data_dir = parent_dir / "data"
dataset_dir = data_dir / "Loan_Default.csv"

# Ensure the project root is in the system path for correct module imports
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))



# -------------------- IMPORT PREPROCESSING FUNC -------------------------

try:
    from preprocess_2 import load_and_split_data, get_preprocessing_pipeline, get_transformed_df
except ImportError:
    print("Error: Could not import preprocessing functions. Ensure 'preprocess_2.py' is correctly set up.")
    sys.exit(1)


# --------------------- MAIN MODELING FUNCTION ----------------------------

def run_logistic_regression(filepath=dataset_dir):
    """
    Executes the full Logistic Regression modeling workflow:
    1. Loads and splits the data.
    2. Builds the full ML pipeline (Preprocessing + Logistic Regression).
    3. Performs GridSearchCV for hyperparameter tuning (C).
    4. Evaluates the best model on the test set.
    5. Visualizes the Confusion Matrix and Feature Importance.
    """
    print(f"--- Starting Logistic Regression Model Training (Data: {filepath.name}) ---")
    
    # --------- LOAD AND SPLIT DATA -------------
    X_train, X_test, y_train, y_test = load_and_split_data(filepath)

    # ---------- PREPROCESSING PIPELINE ----------
    preprocessor = get_preprocessing_pipeline()
    
    # --------- DEFINE FULL ML PIPELINE ----------
    
    # Base Logistic Regression Model (using balanced class weight for imbalanced target)
    model = LogisticRegression(
        solver='liblinear',
        random_state=42,
        class_weight='balanced',
        max_iter=1000 # Increase max_iter for convergence stability
    )

    # Full Pipeline: Preprocessor -> Model
    full_pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor), # This holds the entire data cleaning and transformation
            ('model', model)
        ]
    )
    
    # --------- HYPERPARAMETER TUNING SETUP (Grid Search) -----------

    print("\n--- Starting Grid Search for optimal C parameter ---")
    
    # Define the parameter grid for the 'C' value in Logistic Regression
    param_grid = {
        'model__C': np.logspace(-3, 2, 6) # Generates [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        verbose=1,
        n_jobs=-1 
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_c = grid_search.best_params_['model__C']

    print(f"\n=====================================================================")
    print(f"Optimal C value found: {best_c}")
    print(f"Best cross-validation AUC score: {grid_search.best_score_:.4f}")
    print("=====================================================================")

    # -------- EVALUATION ON TEST SET ------------

    # Predict probabilities on the test set using the best model
    y_pred_proba_test = best_model.predict_proba(X_test)[:, 1]

    # Predict classes on the test set
    y_pred_test = best_model.predict(X_test)

    # Final test set evaluation
    final_auc = roc_auc_score(y_test, y_pred_proba_test)
    print(f"\nFinal Model (Best C={best_c}) AUC Score on Test Set: {final_auc:.4f}")

    print("\n--- Final Classification Report on Test Set ---")
    print(classification_report(y_test, y_pred_test, digits=4))
    
    # ------------------ VISUALIZATION: Normalized Confusion Matrix ------------------
    cm = confusion_matrix(y_test, y_pred_test, normalize='true')
    class_labels = ['0 (No Default)', '1 (Default)']

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=".2%", # Format annotations as percentage with 2 decimal places
        cmap='Blues', 
        xticklabels=class_labels, 
        yticklabels=class_labels,
        cbar=False,
        linewidths=0.5,
        linecolor='black'
    )

    plt.title('Normalized Confusion Matrix (Test Set)', fontsize=16)
    plt.ylabel('Actual Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.savefig(parent_dir / 'images' / 'log_reg_confusion.png') # Save to the images folder
    plt.show()

    print("\nConfusion Matrix Saved: log_reg_confusion.png")
    
    # --------- FEATURE IMPORTANCE ANALYSIS & VISUALIZATION ---------
    
    # 1. Get feature names after all transformations
    # We must call get_transformed_df() to get the processed feature names
    df_processed_train = get_transformed_df()
    
    # The get_transformed_df function already concatenates X_train_trans and y_train.
    # We need to drop the target variable 'Status' to get the feature names.
    features = df_processed_train.drop(columns = 'Status').columns
    
    # 2. Get the coefficients from the fitted Logistic Regression model
    feat_importance = best_model.named_steps['model'].coef_[0]

    # 3. Create DataFrame and sort by absolute coefficient magnitude
    df_feat = pd.DataFrame(
        {
            'Features': features,
            'Coefficient': feat_importance,
            'Absolute_Coefficient': np.abs(feat_importance)
        }
    ).sort_values(by='Absolute_Coefficient', ascending=True)

    # 4. Plotting feature importance (Top 20 absolute values)
    top_n = 20
    df_plot = df_feat.tail(top_n)

    plt.figure(figsize=(10, 10))
    
    # Use different colors for positive and negative coefficients for interpretability
    colors = ['darkred' if c > 0 else 'darkgreen' for c in df_plot['Coefficient']]
    
    df_plot.plot(
        kind='barh', 
        x='Features', 
        y='Coefficient', 
        legend=False, 
        figsize=(10,10),
        color=colors
    )
    plt.title(f'Top {top_n} Features Driving Loan Default Prediction (Logistic Regression)', fontsize=14)
    plt.xlabel('Coefficient Value (Impact on Log-Odds of Default)')
    plt.ylabel('Features')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(parent_dir / 'images' / 'top_20_feat_importance.png') # Save to the images folder
    plt.show()

    print("\nFeature Importance Plot Saved: top_20_feat_importance.png")
    print("\n--- Logistic Regression Workflow Complete ---")
    
   


# -------- EXECUTE MODELING WORKFLOW ------------
if __name__ == '__main__':
    run_logistic_regression()