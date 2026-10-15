import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from common import (
    load_data,
    create_model,
    write_results,
    RANDOM_STATE,
    TEST_SIZE,
)

AGE_CUTOFF = 35
COST_BAD_APPROVED = 5  # Bad applicant approved (FP when 1=good, 0=bad)
COST_GOOD_DENIED = 1   # Good applicant denied  (FN when 1=good, 0=bad)
THRESHOLD = 0.50


def main():
    # 1. Load dataset with readable column names (1 = good credit, 0 = bad credit)
    X, y, _ = load_data(should_remap_column_names=True)

    # 2. Extract sensitive attribute for auditing before dropping
    age_group = np.where(X["age"] < AGE_CUTOFF, "<35", ">=35")

    # 3. Drop "age" from the model's feature set (Fairness through Unawareness)
    X = X.drop(columns=["age"])

    print("Dataset shape after dropping 'age':", X.shape)
    print("Bad-credit rate by age group:")
    for g in ["<35", ">=35"]:
        print(f"  {g}: {(1 - y[age_group == g]).mean():.3f}")

    # 4. Standard 80/20 train/test split stratified by label and age group
    strat = y.astype(str).to_numpy() + "_" + age_group
    (
        X_train,
        X_test,
        y_train,
        y_test,
        age_train,
        age_test,
    ) = train_test_split(
        X,
        y,
        age_group,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=strat,
    )

    # 5. Create and fit model pipeline from common.py
    model = create_model(X_train)
    model.fit(X_train, y_train)

    # 6. Predict on test set using default scikit-learn threshold of 0.50
    y_pred = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]

    # 7. Evaluate overall model performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, test_probs)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    total_cost = int(COST_BAD_APPROVED * fp + COST_GOOD_DENIED * fn)

    print("\n--- Age Fairness Model (Threshold = 0.50) ---")
    print("Shape of test data: ", X_test.shape)
    print("Threshold:          ", THRESHOLD)
    print(f"Accuracy:           {accuracy:.3f}")
    print(f"Precision:          {precision:.3f}")
    print(f"Recall:             {recall:.3f}")
    print(f"F1-score:           {f1:.3f}")
    print(f"ROC-AUC:            {auc:.3f}")
    print(f"Bad applicants approved (FP): {fp}")
    print(f"Good applicants denied  (FN): {fn}")
    print(f"Total error cost:   {total_cost}")

    # 8. Subgroup fairness audit for <35 vs >=35
    groups = {}
    for g in ["<35", ">=35"]:
        mask = age_test == g
        yt = y_test.to_numpy()[mask]
        yp = y_pred[mask]
        g_tn, g_fp, g_fn, g_tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()

        groups[g] = {
            "applicants": int(mask.sum()),
            "approval_rate": float((yp == 1).mean()),
            "fpr_good_denied": float(g_fn / (g_fn + g_tp)) if (g_fn + g_tp) > 0 else 0.0,
            "fnr_bad_approved": float(g_fp / (g_fp + g_tn)) if (g_fp + g_tn) > 0 else 0.0,
            "cost_per_applicant": float(
                (COST_BAD_APPROVED * g_fp + COST_GOOD_DENIED * g_fn) / mask.sum()
            ),
        }

    print("\n--- Per age group (<35 vs >=35) ---")
    for g, m in groups.items():
        print(
            f"{g:>4}: n={m['applicants']}, approval={m['approval_rate']:.3f}, "
            f"Good Denied (FN rate)={m['fpr_good_denied']:.3f}, "
            f"Bad Approved (FP rate)={m['fnr_bad_approved']:.3f}, "
            f"cost/applicant={m['cost_per_applicant']:.3f}"
        )

    # 9. Save combined results dictionary using common.write_results()
    results = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "age_cutoff": AGE_CUTOFF,
        "threshold": THRESHOLD,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "false_positive": int(fp),
        "false_negative": int(fn),
        "total_error_cost": total_cost,
        "age_groups": groups,
    }

    write_results("age_fairness_metrics.json", results)


if __name__ == "__main__":
    main()
