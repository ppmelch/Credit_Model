"""
Configuration module for the credit risk modeling project.

This module centralizes:
- Hyperparameters for different machine learning models
- Project directory paths

This design allows easy modification of model settings and ensures
consistency across the pipeline.
"""

from pathlib import Path


MODEL_CONFIG = {
    "logistic": {
        "max_iter": 1000,
        "solver": "lbfgs"
    },

    "random_forest": {
        "n_estimators": 1000,
        "max_depth": 2,
        "random_state": 42,
        "class_weight": "balanced",
        "n_jobs": -1
    },

    "xgboost": {
        "objective" : "binary:logistic",
        "n_estimators": 250,
        "max_depth": 2,
        "learning_rate": 0.01,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42
    }
}


# Root directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory where trained models are stored
MODELS_DIR = BASE_DIR / "models"
