from pathlib import Path
import json

from common import load_data, run_model


X, y, _ = load_data()

results = run_model(X, y, "Baseline")

project_root = Path(__file__).resolve().parent.parent

results_directory = project_root / "results" / "metrics"
results_directory.mkdir(parents=True, exist_ok=True)

output_file = results_directory / "baseline_metrics.json"

with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to: {output_file}")
