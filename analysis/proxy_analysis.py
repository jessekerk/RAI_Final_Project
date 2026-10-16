#Analysis that tests whether the proxy features are actually proxies for age using: 
# Cramer's V: measures strength of association between categorical variables from 0 to 1 (least to most)
# Spearman association: measures monotonic  association between ordered or numberical variables, from -1 to +1 
# we need to report these measures in the report 

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr

from models.common import (
    AGE_CUTOFF,
    load_data,
    split_data,
)


PROXY_FEATURES = [
    "employment_since",
    "housing",
    "personal_status_sex",
    "residence_since",
]


def cramers_v(x, y):
    """Measure association between two categorical variables."""

    contingency = pd.crosstab(x, y)

    if min(contingency.shape) < 2:
        return 0.0

    chi2 = chi2_contingency(contingency, correction=False).statistic

    n = contingency.to_numpy().sum()
    denominator = n * (min(contingency.shape) - 1)

    return float(np.sqrt(chi2 / denominator))


def main():
    X, y, _ = load_data(should_remap_column_names=True)

    age_groups = np.where(
        X["age"] < AGE_CUTOFF,
        "under_35",
        "35_plus",
    )

    #only training data for proxy identification.
    train_idx, _, _ = split_data(X, y, age_groups)

    X_train = X.iloc[train_idx].copy()
    groups_train = pd.Series(
        age_groups[train_idx],
        index=X_train.index,
        name="age_group",
    )

    results = {}

    for feature in PROXY_FEATURES:
        print(f"\n--- {feature} ---")

        values = X_train[feature]

        if feature == "residence_since":
            correlation, p_value = spearmanr(
                values,
                X_train["age"],
                nan_policy="omit",
            )

            results[feature] = {
                "measure": "spearman_rho",
                "value": (float(correlation) if np.isfinite(correlation) else None),
                "p_value": (float(p_value) if np.isfinite(p_value) else None),
            }

        else:
            association = cramers_v(values, groups_train)

            results[feature] = {
                "measure": "cramers_v",
                "value": association,
            }

            distribution = pd.crosstab(
                values,
                groups_train,
                normalize="columns",
            )

            print("\nDistribution by age group:")
            print(distribution.round(3))

        print("Association:", results[feature])

    output = (
        Path(__file__).resolve().parent.parent
        / "results"
        / "metrics"
        / "proxy_analysis.json"
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
