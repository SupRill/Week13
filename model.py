# model.py
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)


# ====================================================
# UTILITAS: NORMALISASI TARGET
# ====================================================
def normalize_target(series):
    mapping = {
        "true": 1, "True": 1, "TRUE": 1,
        "yes": 1, "Ya": 1, "YA": 1, "Y": 1, "1": 1,
        "false": 0, "False": 0, "FALSE": 0,
        "no": 0, "Tidak": 0, "TIDAK": 0, "N": 0, "0": 0
    }
    if series.dtype == object:
        series = series.map(lambda v: mapping.get(str(v).strip(), v))
    return pd.to_numeric(series, errors="coerce")


# ====================================================
# PIPELINE BUILDER
# ====================================================
def build_pipeline(algorithm, numeric_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="passthrough"
    )

    if algorithm == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = SVC(kernel="rbf", probability=True)

    pipeline = Pipeline(steps=[
        ("pre", preprocessor),
        ("model", model)
    ])

    return pipeline


# ====================================================
# HYPERPARAMETER GRIDS
# ====================================================
def get_param_grid(algorithm):
    if algorithm == "Logistic Regression":
        return {
            "model__C": [0.01, 0.1, 1, 10],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs"],
            "model__max_iter": [200]
        }
    else:
        return {
            "model__C": [0.1, 1, 10],
            "model__gamma": ["scale", "auto"],
            "model__probability": [True]
        }


# ====================================================
# TRAINING FUNCTION
# ====================================================
def train_model(df, features, target, algorithm="Logistic Regression",
                test_size=0.2, random_state=42, tuning=True):
    
    X = df[features]
    y = normalize_target(df[target])

    # Handle NaN in target
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    pipeline = build_pipeline(algorithm, numeric_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y if y.nunique() > 1 else None,
        random_state=random_state
    )

    if tuning:
        param_grid = get_param_grid(algorithm)
        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        best_params = grid.best_params_
    else:
        model = pipeline
        model.fit(X_train, y_train)
        best_params = None

    # Predictions
    y_pred = model.predict(X_test)

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except:
        y_scores = model.decision_function(X_test)
        y_proba = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-9)

    # Metrics
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "best_params": best_params,
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba
    }

    return results


# ====================================================
# SAVE & LOAD MODEL
# ====================================================
def save_model(model, filename="model.joblib"):
    joblib.dump(model, filename)
    return filename


def load_model(filename):
    return joblib.load(filename)


# ====================================================
# PREDICTION UTILITY
# ====================================================
def predict_dataframe(model, df, features):
    X = df[features]
    preds = model.predict(X)
    try:
        proba = model.predict_proba(X)[:, 1]
    except:
        proba = None
    return preds, proba
