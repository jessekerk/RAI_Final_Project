import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


MIN_GROUP_SIZE = 10
MIN_LABEL_COUNT = 5


def evaluate_intersectional(
    y_true,
    y_pred,
    age_groups,
    personal_status,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    age_groups = np.asarray(age_groups)
    personal_status = np.asarray(personal_status)

    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "age_group": age_groups,
            "personal_status_sex": personal_status,
        }
    )

    results = {}

    for (age, status), group in df.groupby(["age_group", "personal_status_sex"]):
        key = f"{age}_{status}"

        n = len(group)
        good_count = int((group["y_true"] == 1).sum())
        bad_count = int((group["y_true"] == 0).sum())

        # Only interpret rates when support is sufficient.
        sufficient = (
            n >= MIN_GROUP_SIZE
            and good_count >= MIN_LABEL_COUNT
            and bad_count >= MIN_LABEL_COUNT
        )

        result = {
            "applicants": n,
            "good_credit": good_count,
            "bad_credit": bad_count,
            "sufficient_support": sufficient,
        }

        if sufficient:
            yt = group["y_true"]
            yp = group["y_pred"]

            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()

            result.update(
                {
                    "approval_rate": float((yp == 1).mean()),
                    "true_positive_rate": float(tp / (tp + fn)),
                    "false_positive_rate": float(fp / (fp + tn)),
                    "false_negative_rate": float(fn / (fn + tp)),
                    "cost_per_applicant": float((5 * fp + fn) / n),
                }
            )

        results[key] = result

    return results
