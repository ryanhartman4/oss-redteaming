from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

import src.small_model_redteaming.eval_awareness as eva
import src.small_model_redteaming.set_up.model_utils as mu
import os
from dotenv import load_dotenv

load_dotenv()
import json
import pandas as pd
import numpy as np
from scipy import stats

log_probs_model = mu.PreGenAnalyzer(api_key=os.getenv("API_KEY"))

# Store results for comparison
all_results = {}
test_conditions = [
    "baseline",
    "explicit",
    "high_stakes_evaluation",
    "leet_speak",
    "fictional",
]

# Run each test twice
for run in [1, 2]:
    print(f"RUN {run}")

    for condition in test_conditions:
        print(f"\nRunning {condition}...")
        if condition == "baseline":
            condition_model = mu.PreGenAnalyzer(api_key=os.getenv("API_KEY"))
        else:
            condition_model = mu.PreGenAnalyzer(
                api_key=os.getenv("API_KEY"), test_condition=condition
            )

        tester = eva.AwarenessImpactTest(condition_model)
        results = tester.run_mmlu_test(limit=50, subjects=["mmlu_miscellaneous"])

        if condition not in all_results:
            all_results[condition] = []
        all_results[condition].append(results["overall_accuracy"])

        print(f"Overall Accuracy: {results['overall_accuracy']*100:.1f}%")

# Statistical comparison
print("STATISTICAL COMPARISON WITH BASELINE")

baseline_accuracies = all_results["baseline"]
baseline_mean = np.mean(baseline_accuracies)

for condition in test_conditions:
    if condition != "baseline":
        condition_accuracies = all_results[condition]
        condition_mean = np.mean(condition_accuracies)

        try:
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(baseline_accuracies, condition_accuracies)

            print(
                f"{condition} vs baseline: "
                f"mean={condition_mean:.3f} vs {baseline_mean:.3f}, "
                f"t={t_stat:.3f}, p={p_value:.3f}"
            )

            if p_value < 0.1:
                direction = "higher" if condition_mean > baseline_mean else "lower"
                print(f"Significantly {direction} accuracy (p < 0.1)")
            else:
                print(f"No significant difference (p ≥ 0.1)")
        except Exception as e:
            print(f"Exception occurred for {condition}, Exception: {e}")
