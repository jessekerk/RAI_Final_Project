from pathlib import Path
import json

import pandas as pd

from common import load_data, run_model

from models.common import remap_column_names

REMOVE_IN_VARIANT: list[list[str]] = [
    ['age', 'employment_since'],
    ['age', 'housing'],
    ['age', 'personal_status_sex'],
    ['age', 'residence_since'],
    ['age', 'employment_since', 'housing', 'personal_status_sex', 'residence_since'],
]


def run_for_variant(X_variant: pd.DataFrame, y: pd.DataFrame, remove_in_variant: list[str]) -> None:
    variant_name = "_".join(remove_in_variant)

    results = run_model(X_variant, y, f"Proxy Reduced {variant_name}")

    project_root = Path(__file__).resolve().parent.parent

    results_directory = project_root / "results" / "metrics"
    results_directory.mkdir(parents=True, exist_ok=True)

    output_file = results_directory / f"proxy_reduced_{variant_name}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {output_file}")



def main():
    X, y, _ = load_data()

    X = remap_column_names(X)

    for remove_in_variant in REMOVE_IN_VARIANT:
        X_variant = X.drop(columns=remove_in_variant)
        run_for_variant(X_variant, y, remove_in_variant)


if __name__ == "__main__":
    main()

