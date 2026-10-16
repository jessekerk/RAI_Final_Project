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


def main():
    X, y, _ = load_data(should_remap_column_names=True)

    #age retained for fairness auditing
    age_groups = np.where(
        X["age"] < AGE_CUTOFF,
        "under_35",
        "35_plus",
    )

    #split once for all models
    train_idx, val_idx, test_idx = split_data(X, y, age_groups)

    all_results = {}

    for model_name, removed_features in MODELS.items():
        print(f"\n{'=' * 50}")
        print(f"MODEL: {model_name}")
        print(f"{'=' * 50}")

        #remove features for this model
        X_model = X.drop(columns=removed_features)

        #use identical row indices for all models
        X_train = X_model.iloc[train_idx]
        X_val = X_model.iloc[val_idx]
        X_test = X_model.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        y_test = y.iloc[test_idx]

        #create and train model
        model = create_model(X_train)
        model.fit(X_train, y_train)

        #select threshold on validation data
        threshold = select_threshold(model, X_val, y_val)

        #predict on test set
        probabilities = model.predict_proba(X_test)[:, 1]

        y_pred = (probabilities >= threshold).astype(int)

        #standard performance evaluation
        performance = evaluate_performance(
            y_test,
            y_pred,
            probabilities,
        )

        #fairness eval
        fairness = evaluate_fairness(
            y_test,
            y_pred,
            age_groups[test_idx],
        )

        #intersectional analysis
        intersectional = evaluate_intersectional(
            y_test,
            y_pred,
            age_groups[test_idx],
            X.iloc[test_idx]["personal_status_sex"],
        )

        #combine results
        results = {
            "model": model_name,
            "removed_features": removed_features,
            "random_state": RANDOM_STATE,
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

        #individual model results
        write_results(
            f"{model_name}_metrics.json",
            results,
        )

        #performance
        print(f"\nThreshold: {threshold:.2f}")
        print(f"Accuracy: {performance['accuracy']:.3f}")
        print(f"Precision: {performance['precision']:.3f}")
        print(f"Recall: {performance['recall']:.3f}")
        print(f"F1-score: {performance['f1']:.3f}")
        print(f"ROC-AUC: {performance['roc_auc']:.3f}")

        print(f"Total error cost: {performance['total_error_cost']}")

        #fairness
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

    #results
    write_results(
        "all_models_metrics.json",
        all_results,
    )

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
