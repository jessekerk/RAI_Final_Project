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
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
AGE_CUTOFF = 35
COST_FN = 5  # bad applicant approved
COST_FP = 1  # good applicant denied


def total_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return COST_FN * fn + COST_FP * fp


#load dataset

german_credit = fetch_ucirepo(id=144)

X = german_credit.data.features.copy()
y = german_credit.data.targets.iloc[:, 0].copy()

# 0 = bad credit, 1 = good credit)
y = y.map({1: 1, 2: 0})

# age is the sensitive attribute: used only for auditing, never as a model input
age_group = np.where(X["Attribute13"] < AGE_CUTOFF, "<35", ">=35")
age_group = np.asarray(age_group)
X = X.drop(columns=["Attribute13"])

print("Dataset shape:", X.shape)
print("Bad-credit rate by age group:")
for g in ["<35", ">=35"]:
    print(f"  {g}: {y[age_group == g].mean():.3f}")


#feature types

categorical_features = X.select_dtypes(include=["object", "str", "category"]).columns.tolist()
numerical_features = X.select_dtypes(include=["number"]).columns.tolist()


#train / validation / test split (80/10/10, stratified by label + age group)

strat = y.astype(str).to_numpy() + "_" + age_group

X_train, X_tmp, y_train, y_tmp, age_train, age_tmp, _, strat_tmp = train_test_split(
    X, y, age_group, strat, test_size=0.20, random_state=RANDOM_STATE, stratify=strat
)
X_val, X_test, y_val, y_test, age_val, age_test = train_test_split(
    X_tmp, y_tmp, age_tmp, test_size=0.50, random_state=RANDOM_STATE, stratify=strat_tmp
)


#model (same pipeline as baseline)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numerical_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]
)
model.fit(X_train, y_train)


#choose the cost-minimising threshold on the validation set

val_probs = model.predict_proba(X_val)[:, 1]
thresholds = np.arange(0.05, 0.95, 0.01)
val_costs = [total_cost(y_val, (val_probs >= t).astype(int)) for t in thresholds]
threshold = round(float(thresholds[int(np.argmin(val_costs))]), 2)

print("\nChosen threshold (validation):", threshold)


#evaluate on the test set

test_probs = model.predict_proba(X_test)[:, 1]
y_pred = (test_probs >= threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, test_probs)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
cost = int(COST_FN * fn + COST_FP * fp)

print("\n--- Age Model (age excluded from features) ---")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")
print(f"ROC-AUC:   {auc:.3f}")
print(f"Bad applicants approved (FN): {fn}")
print(f"Good applicants denied  (FP): {fp}")
print(f"Total cost: {cost}")


#fairness audit per age group

groups = {}
for g in ["<35", ">=35"]:
    mask = age_test == g
    yt, yp = y_test.to_numpy()[mask], y_pred[mask]
    g_tn, g_fp, g_fn, g_tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    groups[g] = {
        "applicants": int(mask.sum()),
        "approval_rate": float((yp == 0).mean()),
        "fpr_good_denied": float(g_fp / (g_fp + g_tn)),
        "fnr_bad_approved": float(g_fn / (g_fn + g_tp)),
        "cost_per_applicant": float((COST_FN * g_fn + COST_FP * g_fp) / mask.sum()),
    }

print("\n--- Per age group ---")
for g, m in groups.items():
    print(
        f"{g:>4}: n={m['applicants']}, approval={m['approval_rate']:.3f}, "
        f"FPR={m['fpr_good_denied']:.3f}, FNR={m['fnr_bad_approved']:.3f}, "
        f"cost/applicant={m['cost_per_applicant']:.3f}"
    )


#save metrics
results = {
    "random_state": RANDOM_STATE,
    "age_cutoff": AGE_CUTOFF,
    "threshold": threshold,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": auc,
    "false_positive": int(fp),
    "false_negative": int(fn),
    "total_error_cost": cost,
    "age_groups": groups,
}

project_root = Path(__file__).resolve().parent
results_directory = project_root / "results" / "metrics"
results_directory.mkdir(parents=True, exist_ok=True)

output_file = results_directory / "age_model_metrics.json"

with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to: {output_file}")