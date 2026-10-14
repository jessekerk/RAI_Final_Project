import json
from pathlib import Path
from typing import Tuple, Any

import numpy as np
import pandas
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

RANDOM_STATE = 42
TEST_SIZE = 0.20

COLUMN_NAMES: list[str] = [
    "checking_status",      # Attribute 1
    "duration_months",      # Attribute 2
    "credit_history",       # Attribute 3
    "purpose",              # Attribute 4
    "credit_amount",        # Attribute 5
    "savings",              # Attribute 6
    "employment_since",     # Attribute 7
    "installment_rate",     # Attribute 8
    "personal_status_sex",  # Attribute 9
    "other_debtors",        # Attribute 10
    "residence_since",      # Attribute 11
    "property",             # Attribute 12
    "age",                  # Attribute 13
    "other_installment",    # Attribute 14
    "housing",              # Attribute 15
    "existing_credits",     # Attribute 16
    "job",                  # Attribute 17
    "people_liable",        # Attribute 18
    "telephone",            # Attribute 19
    "foreign_worker",       # Attribute 20
]


def load_data() -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    # load dateset

    german_credit = fetch_ucirepo(id=144)

    X = german_credit.data.features.copy()
    y = german_credit.data.targets.iloc[:, 0].copy()

    print("Dataset shape:", X.shape)
    print("\nOriginal target distribution:")
    print(y.value_counts())

    # convert good/bad credit score
    y = y.map({1: 1, 2: 0})

    return X, y, german_credit.variables

def remap_column_names(X: pandas.DataFrame) -> pandas.DataFrame:
    mapping = {
        f"Attribute{i}": name
        for i, name in enumerate(COLUMN_NAMES, start=1)
    }
    return X.rename(columns=mapping, errors="raise")


def identify_features(X: pandas.DataFrame) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    # identify categorical/numerical features

    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numerical_features = X.select_dtypes(include=["number"]).columns.tolist()

    print("\nCategorical features:")
    print(categorical_features)

    print("\nNumerical features:")
    print(numerical_features)

    return categorical_features, numerical_features

def split_data(X: pandas.DataFrame, y: pandas.DataFrame) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test

def create_model(X: pandas.DataFrame) -> Pipeline:
    # identify categorical/numerical features

    categorical_features, numerical_features = identify_features(X)

    # preprocessing

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numerical_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    # log-reg

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )

    return model

def run_model(X: pd.DataFrame, y: pd.DataFrame, model_name: str) -> dict[str, int | float | Any]:
    # train/test slit

    X_train, X_test, y_train, y_test = split_data(X, y)

    # Create model

    model = create_model(X_train)

    # Train model

    model.fit(X_train, y_train)

    # predictions

    y_pred = model.predict(X_test)

    # eval

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n--- {model_name} Logistic Regression ---")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Bad credit", "Good credit"]))

    # cost-sensitive eval
    # In German Credit:
    # bad applicant predicted as good = cost 5
    # good applicant predicted as bad = cost 1

    false_positive = np.sum((y_test == 0) & (y_pred == 1))

    false_negative = np.sum((y_test == 1) & (y_pred == 0))

    total_cost = 5 * false_positive + false_negative

    print("\n--- Error Costs ---")
    print("Bad applicants predicted as good:", false_positive)
    print("Good applicants predicted as bad:", false_negative)
    print("Total misclassification cost:", total_cost)

    # save metrics
    results = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "total_error_cost": int(total_cost),
    }

    return results

def write_results(file_name: str, results: dict[str, int | float | Any]) -> None:
    project_root = Path(__file__).resolve().parent.parent

    results_directory = project_root / "results" / "metrics"
    results_directory.mkdir(parents=True, exist_ok=True)

    output_file = results_directory / file_name

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {output_file}")
