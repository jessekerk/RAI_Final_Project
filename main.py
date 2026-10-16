import numpy as np
from results.intersectional import evaluate_intersectional
from models.common import (
    AGE_CUTOFF,
    RANDOM_STATE,
    load_data,
    split_data,
    create_model,
    select_threshold,
    evaluate_performance,
    write_results,
)
from results.fairness import evaluate_fairness


SEEDS = list(range(42, 52))

MODELS = {
    "baseline": [],
    "age_unaware": [
        "age",
    ],
    "proxy_reduced": [
        "age",
        "employment_since",
        "housing",
        "personal_status_sex",
        "residence_since",
    ],
}

METRICS = [
    ("performance", "accuracy"),
    ("performance", "precision"),
    ("performance", "recall"),
    ("performance", "f1"),
    ("performance", "roc_auc"),
    ("performance", "total_error_cost"),
    ("fairness", "demographic_parity_difference"),
    ("fairness", "equal_opportunity_difference"),
]


def run_experiment(X, y, age_groups, seed):
    # split once for all models
    train_idx, val_idx, test_idx = split_data(X, y, age_groups, random_state=seed)

    all_results = {}

    for model_name, removed_features in MODELS.items():
        print(f"\n{'=' * 50}")
        print(f"MODEL: {model_name} | SEED: {seed}")
        print(f"{'=' * 50}")

        # remove features for this model
        X_model = X.drop(columns=removed_features)

        # use identical row indices for all models
        X_train = X_model.iloc[train_idx]
        X_val = X_model.iloc[val_idx]
        X_test = X_model.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        y_test = y.iloc[test_idx]

        # create and train model
        model = create_model(X_train)
        model.fit(X_train, y_train)

        # select threshold on validation data
        threshold = select_threshold(model, X_val, y_val)

        # predict on test set
        probabilities = model.predict_proba(X_test)[:, 1]

        y_pred = (probabilities >= threshold).astype(int)

        # standard performance evaluation
        performance = evaluate_performance(
            y_test,
            y_pred,
            probabilities,
        )

        # fairness eval
        fairness = evaluate_fairness(
            y_test,
            y_pred,
            age_groups[test_idx],
        )

        # intersectional analysis
        intersectional = evaluate_intersectional(
            y_test,
            y_pred,
            age_groups[test_idx],
            X.iloc[test_idx]["personal_status_sex"],
        )

        # combine results
        results = {
            "model": model_name,
            "removed_features": removed_features,
            "random_state": seed,
            "age_cutoff": AGE_CUTOFF,
            "train_size": len(train_idx),
            "validation_size": len(val_idx),
            "test_size": len(test_idx),
            "threshold": threshold,
            "performance": performance,
            "fairness": fairness,
            "intersectional": intersectional,
        }

        all_results[model_name] = results

        # individual model results for original seed
        if seed == RANDOM_STATE:
            write_results(
                f"{model_name}_metrics.json",
                results,
            )

        # performance
        print(f"\nThreshold: {threshold:.2f}")
        print(f"Accuracy: {performance['accuracy']:.3f}")
        print(f"Precision: {performance['precision']:.3f}")
        print(f"Recall: {performance['recall']:.3f}")
        print(f"F1-score: {performance['f1']:.3f}")
        print(f"ROC-AUC: {performance['roc_auc']:.3f}")

        print(f"Total error cost: {performance['total_error_cost']}")

        # fairness
        print("\n--- Fairness ---")

        print(
            "Demographic parity difference:",
            round(
                fairness["demographic_parity_difference"],
                3,
            ),
        )

        eo = fairness["equal_opportunity_difference"]

        print(
            "Equal opportunity difference:",
            round(eo, 3) if eo is not None else "Undefined",
        )

        for group, metrics in fairness["age_groups"].items():
            print(f"\nAge group: {group}")
            print(f"Applicants: {metrics['applicants']}")
            print(f"Approval rate: {metrics['approval_rate']:.3f}")
            print(f"Cost per applicant: {metrics['cost_per_applicant']:.3f}")

    return all_results


def summarize_results(all_results):
    # calculate mean and standard deviation across seeds
    summary = {}

    for model_name in MODELS:
        summary[model_name] = {}

        for category, metric in METRICS:
            values = [
                all_results[str(seed)][model_name][category][metric] for seed in SEEDS
            ]

            # ignore undefined metrics
            values = np.array(
                [value for value in values if value is not None],
                dtype=float,
            )

            if len(values) == 0:
                summary[model_name][metric] = {
                    "mean": None,
                    "std": None,
                    "n": 0,
                }
                continue

            summary[model_name][metric] = {
                "mean": float(np.mean(values)),
                "std": (float(np.std(values, ddof=1)) if len(values) > 1 else None),
                "n": int(len(values)),
            }

    return summary


def print_summary(summary):
    # print mean and standard deviation across seeds
    print("\n" + "=" * 60)
    print("SUMMARY ACROSS RANDOM SEEDS")
    print("=" * 60)

    for model_name, metrics in summary.items():
        print(f"\n--- {model_name} ---")

        for metric, values in metrics.items():
            mean = values["mean"]
            std = values["std"]

            if mean is None:
                print(f"{metric:35s}: Undefined")
            elif std is None:
                print(f"{metric:35s}: {mean:.3f}")
            else:
                print(f"{metric:35s}: {mean:.3f} +/- {std:.3f}")


def main():
    X, y, _ = load_data(should_remap_column_names=True)

    # age retained for fairness auditing
    age_groups = np.where(
        X["age"] < AGE_CUTOFF,
        "under_35",
        "35_plus",
    )

    all_results = {}

    # run all models across different random seeds
    for seed in SEEDS:
        print(f"\n{'=' * 60}")
        print(f"RANDOM SEED: {seed}")
        print(f"{'=' * 60}")

        results = run_experiment(X, y, age_groups, seed)

        all_results[str(seed)] = results

        # results for original seed
        if seed == RANDOM_STATE:
            write_results(
                "all_models_metrics.json",
                results,
            )

    # calculate summary statistics
    summary = summarize_results(all_results)

    # results
    write_results(
        "repeated_seeds.json",
        all_results,
    )

    write_results(
        "summary_metrics.json",
        summary,
    )

    print_summary(summary)

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
