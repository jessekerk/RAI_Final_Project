import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ucimlrepo import fetch_ucirepo


RANDOM_STATE = 42
AGE_CUTOFF = 35

# Target encoding:
# 1 = good credit, 0 = bad credit
COST_FALSE_APPROVAL = 5
COST_FALSE_REJECTION = 1


COLUMN_NAMES = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment",
    "housing",
    "existing_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
]


def load_data(should_remap_column_names=True):
    german_credit = fetch_ucirepo(id=144)

    X = german_credit.data.features.copy()
    y = german_credit.data.targets.iloc[:, 0].copy()

    # 1 = good credit, 0 = bad credit
    y = y.map({1: 1, 2: 0}).astype(int)

    if should_remap_column_names:
        X = remap_column_names(X)

    return X, y, german_credit.variables


def remap_column_names(X):
    mapping = {f"Attribute{i}": name for i, name in enumerate(COLUMN_NAMES, start=1)}

    return X.rename(columns=mapping, errors="raise")


def identify_features(X):
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(include=["number"]).columns.tolist()

    return categorical_features, numerical_features


def split_data(X, y, age_groups):
    """
    Create a shared 80/10/10 train/validation/test split.
    Stratify by both target and age group to help
    preserve subgroup representation.
    """

    indices = np.arange(len(X))
    age_groups = np.asarray(age_groups)

    stratification = y.astype(str).to_numpy() + "_" + age_groups

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=stratification,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=stratification[temp_idx],
    )

    return train_idx, val_idx, test_idx


def create_model(X):
    categorical_features, numerical_features = identify_features(X)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numerical_features,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return model


def total_cost(y_true, y_pred):
    """
    German Credit misclassification costs:

    Bad applicant predicted good: cost 5
    Good applicant predicted bad: cost 1
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_approvals = np.sum((y_true == 0) & (y_pred == 1))

    false_rejections = np.sum((y_true == 1) & (y_pred == 0))

    return int(
        COST_FALSE_APPROVAL * false_approvals + COST_FALSE_REJECTION * false_rejections
    )


def select_threshold(model, X_val, y_val):
    """
    Select the threshold minimizing classification cost
    on the validation set.
    """

    probabilities = model.predict_proba(X_val)[:, 1]

    thresholds = np.arange(0.05, 0.96, 0.01)

    costs = [
        total_cost(
            y_val,
            (probabilities >= threshold).astype(int),
        )
        for threshold in thresholds
    ]

    best_index = int(np.argmin(costs))

    return round(float(thresholds[best_index]), 2)


def evaluate_performance(y_true, y_pred, probabilities):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "false_approvals": int(fp),
        "false_rejections": int(fn),
        "true_negatives": int(tn),
        "true_positives": int(tp),
        "total_error_cost": total_cost(y_true, y_pred),
    }


def write_results(file_name, results):
    project_root = Path(__file__).resolve().parent.parent

    results_directory = project_root / "results" / "metrics"
    results_directory.mkdir(parents=True, exist_ok=True)

    output_file = results_directory / file_name

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to: {output_file}")
