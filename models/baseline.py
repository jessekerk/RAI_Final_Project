from pathlib import Path
import json

import numpy as np

from ucimlrepo import fetch_ucirepo

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


RANDOM_STATE = 42
TEST_SIZE = 0.20


#load dateset

german_credit = fetch_ucirepo(id=144)

X = german_credit.data.features.copy()
y = german_credit.data.targets.iloc[:, 0].copy()

print("Dataset shape:", X.shape)
print("\nOriginal target distribution:")
print(y.value_counts())


#convert good/bad credit score
y = y.map({1: 1, 2: 0})


#identify categorical/numerical features

categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

numerical_features = X.select_dtypes(include=["number"]).columns.tolist()

print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


#train/test slit

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)


#preprocessing

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


#log-reg

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]
)

model.fit(X_train, y_train)


#predictions

y_pred = model.predict(X_test)


#eval

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

print("\n--- Baseline Logistic Regression ---")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")

print("\nConfusion matrix:")
print(cm)

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Bad credit", "Good credit"]))


#cost-sensitive eval
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


#save metrics
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

project_root = Path(__file__).resolve().parent.parent

results_directory = project_root / "results" / "metrics"
results_directory.mkdir(parents=True, exist_ok=True)

output_file = results_directory / "baseline_metrics.json"

with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to: {output_file}")
