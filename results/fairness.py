import numpy as np
from sklearn.metrics import confusion_matrix


def evaluate_fairness(y_true, y_pred, age_groups):
    """
    Evaluate fairness across two age groups. 1 = good credit, 0 = bad credit
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    age_groups = np.asarray(age_groups)

    results = {}

    for group in np.unique(age_groups):
        mask = age_groups == group

        yt = y_true[mask]
        yp = y_pred[mask]

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()

        approval_rate = np.mean(yp == 1)

        tpr = tp / (tp + fn) if tp + fn > 0 else None

        fpr = fp / (fp + tn) if fp + tn > 0 else None

        fnr = fn / (fn + tp) if fn + tp > 0 else None

        group_cost = 5 * fp + fn

        results[str(group)] = {
            "applicants": int(mask.sum()),
            "approval_rate": float(approval_rate),
            "true_positive_rate": (float(tpr) if tpr is not None else None),
            "false_positive_rate": (float(fpr) if fpr is not None else None),
            "false_negative_rate": (float(fnr) if fnr is not None else None),
            "cost_per_applicant": float(group_cost / mask.sum()),
        }

    group_names = list(results.keys())

    if len(group_names) != 2:
        raise ValueError("Fairness evaluation requires exactly two age groups.")

    group_a = results[group_names[0]]
    group_b = results[group_names[1]]

    # Demographic parity difference
    dp_difference = abs(group_a["approval_rate"] - group_b["approval_rate"])

    # Equal opportunity difference
    tpr_a = group_a["true_positive_rate"]
    tpr_b = group_b["true_positive_rate"]

    eo_difference = (
        abs(tpr_a - tpr_b) if tpr_a is not None and tpr_b is not None else None
    )

    return {
        "age_groups": results,
        "demographic_parity_difference": float(dp_difference),
        "equal_opportunity_difference": (
            float(eo_difference) if eo_difference is not None else None
        ),
    }
