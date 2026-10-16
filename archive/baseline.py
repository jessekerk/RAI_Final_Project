from common import load_data, run_model, write_results

X, y, _ = load_data()

results = run_model(X, y, "Baseline")

write_results("baseline_metrics.json", results)
