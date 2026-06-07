import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE

# Argument parser for MLflow Project
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, default=100)
parser.add_argument('--max_depth', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=0.05)
args = parser.parse_args()

# MLflow configuration via environment variables
mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI'))
mlflow.set_experiment("Diabetes_Prediction_CI")


def load_data():
    """Load train and test datasets."""
    train = pd.read_csv('diabetes_train.csv')
    test = pd.read_csv('diabetes_test.csv')

    X_train = train.drop('Diabetes_binary', axis=1)
    y_train = train['Diabetes_binary']
    X_test = test.drop('Diabetes_binary', axis=1)
    y_test = test['Diabetes_binary']

    print(f"Train shape : {X_train.shape}")
    print(f"Test shape  : {X_test.shape}")
    print(f"Class distribution (train): {y_train.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    """Balance training data using SMOTE."""
    print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE : {pd.Series(y_bal).value_counts().to_dict()}")
    return X_bal, y_bal


def train_model(X_train, X_test, y_train, y_test, model, model_name):
    """Train model with MLflow logging."""
    print(f"\nTraining {model_name}...")

    with mlflow.start_run(run_name=model_name):

        if model_name == "XGBoost":
            params = {
                "n_estimators"    : model.get_params()['n_estimators'],
                "max_depth"       : model.get_params()['max_depth'],
                "learning_rate"   : model.get_params()['learning_rate'],
                "subsample"       : model.get_params()['subsample'],
                "colsample_bytree": model.get_params()['colsample_bytree'],
                "eval_metric"     : model.get_params()['eval_metric'],
                "random_state"    : model.get_params()['random_state'],
            }
            mlflow.log_params(params)
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(
                model, "model",
                input_example=X_test.iloc[:5]
            )
        else:
            mlflow.sklearn.autolog()
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy" : accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall"   : recall_score(y_test, y_pred),
            "f1_score" : f1_score(y_test, y_pred),
            "auc_roc"  : roc_auc_score(y_test, y_prob)
        }

        mlflow.log_metrics(metrics)

        for name, value in metrics.items():
            print(f"  {name:<12}: {value:.4f}")

    print(f"{model_name} - training complete.")


if __name__ == "__main__":
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    print("\nApplying SMOTE...")
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    print("\nStarting model training...")

    models = [
        (LogisticRegression(random_state=42, max_iter=1000), "Logistic_Regression"),
        (RandomForestClassifier(random_state=42, n_estimators=100), "Random_Forest"),
        (XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate
        ), "XGBoost")
    ]

    for model, name in models:
        train_model(X_train_bal, X_test, y_train_bal, y_test, model, name)

    print("\nAll models trained successfully.")
    print("Results are available on DagsHub MLflow dashboard.")