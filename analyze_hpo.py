#!/usr/bin/env python3
"""Analyze Optuna HPO results and save visualizations."""

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_contour,
    plot_slice,
)
import os

RESULTS_DIR = "hpo_results"
storage = f"sqlite:///{RESULTS_DIR}/hpo.db"
study = optuna.load_study(study_name="skullking-hpo", storage=storage)

# Print summary
print(f"Number of finished trials: {len(study.trials)}")
print(f"\nBest trial (#{study.best_trial.number}):")
print(f"  Value: {study.best_trial.value}")
print(f"  Params:")
for k, v in study.best_params.items():
    print(f"    {k}: {v}")

print("\nParameter importances:")
importances = optuna.importance.get_param_importances(study)
for k, v in importances.items():
    print(f"  {k}: {v:.4f}")

# Save visualizations
plots = {
    "optimization_history": plot_optimization_history,
    "param_importances": plot_param_importances,
    "parallel_coordinate": plot_parallel_coordinate,
    "contour": plot_contour,
    "slice": plot_slice,
}

for name, plot_fn in plots.items():
    path = os.path.join(RESULTS_DIR, f"{name}.html")
    try:
        fig = plot_fn(study)
        fig.write_html(path)
        print(f"Saved {path}")
    except Exception as e:
        print(f"Skipped {name}: {e}")

# Also save trials dataframe
df = study.trials_dataframe()
csv_path = os.path.join(RESULTS_DIR, "trials.csv")
df.to_csv(csv_path, index=False)
print(f"Saved {csv_path}")
