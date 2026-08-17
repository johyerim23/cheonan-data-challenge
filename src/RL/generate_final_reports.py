"""Generate comparable model and final policy before/after artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from policy_actions import (
    apply_policy,
    load_ambulance_increment_effects,
    policy_reward_bonus,
)
from simulator import EmergencyPolicySimulator


METRICS = [
    "mean_ems_time", "mean_hospital_time", "mean_total_time",
    "weighted_ems_time", "weighted_hospital_time", "weighted_total_time",
    "max_ems_time", "max_hospital_time",
]


def build_reports(data_dir, results_dir):
    results_dir = Path(results_dir)
    random_rows = pd.read_csv(results_dir / "random_episodes.csv")
    rule = pd.read_csv(results_dir / "rule_summary.csv").iloc[0]
    dqn_rows = pd.read_csv(results_dir / "dqn_seed_evaluation.csv")

    comparison = pd.DataFrame(
        [
            {
                "method": "random_1000",
                "evaluation_units": len(random_rows),
                "reward_mean": random_rows["reward"].mean(),
                "reward_std": random_rows["reward"].std(ddof=1),
                "reward_min": random_rows["reward"].min(),
                "reward_max": random_rows["reward"].max(),
                **{metric: random_rows[metric].mean() for metric in METRICS},
                "dispatch_response_improvement_min": random_rows[
                    "dispatch_response_improvement_min"
                ].mean(),
            },
            {
                "method": "rule_greedy",
                "evaluation_units": 1,
                "reward_mean": rule["reward"],
                "reward_std": 0.0,
                "reward_min": rule["reward"],
                "reward_max": rule["reward"],
                **{metric: rule[metric] for metric in METRICS},
                "dispatch_response_improvement_min": rule[
                    "dispatch_response_improvement_min"
                ],
            },
            {
                "method": "dqn_50000_multi_seed",
                "evaluation_units": len(dqn_rows),
                "reward_mean": dqn_rows["reward"].mean(),
                "reward_std": dqn_rows["reward"].std(ddof=1),
                "reward_min": dqn_rows["reward"].min(),
                "reward_max": dqn_rows["reward"].max(),
                **{metric: dqn_rows[metric].mean() for metric in METRICS},
                "dispatch_response_improvement_min": dqn_rows[
                    "dispatch_response_improvement_min"
                ].mean(),
            },
        ]
    )

    best_dqn = dqn_rows.loc[dqn_rows["reward"].idxmax()]
    if float(rule["reward"]) >= float(best_dqn["reward"]):
        source_method = "rule_greedy"
        source_seed = None
        source_model = None
        expected_reward = float(rule["reward"])
        policies = json.loads(rule["selected_policies"])
    else:
        source_method = "dqn_50000_best_seed"
        source_seed = int(best_dqn["seed"])
        source_model = best_dqn["model_name"]
        expected_reward = float(best_dqn["reward"])
        policies = json.loads(best_dqn["selected_policies"])

    sim = EmergencyPolicySimulator(data_dir=data_dir)
    baseline_state = sim.state.copy(deep=True)
    baseline_metrics = sim.calculate_metrics()
    increment_effects = load_ambulance_increment_effects(data_dir)
    dispatch_improvement = 0.0
    for policy in policies:
        apply_policy(sim, policy)
        dispatch_improvement += policy_reward_bonus(policy, increment_effects)
    final_metrics = sim.calculate_metrics()
    calculated_reward = (
        baseline_metrics["weighted_total_time"]
        - final_metrics["weighted_total_time"]
        + dispatch_improvement
    )
    if not np.isclose(calculated_reward, expected_reward, atol=1e-8):
        raise AssertionError(
            f"선택 정책 보상 불일치: source={expected_reward}, calculated={calculated_reward}"
        )

    before_after = baseline_state[["region_id", "region_name"]].copy()
    columns = [
        "ems_duration_min", "hospital_duration_min", "nearest_alternative_time_min",
        "alternative_gap_min", "alternative_score_01", "vulnerability_type_count",
        "vulnerability_types",
    ]
    for column in columns:
        before_after[f"before_{column}"] = baseline_state[column].to_numpy()
        before_after[f"after_{column}"] = sim.state[column].to_numpy()
    before_after["ems_improvement_min"] = (
        before_after["before_ems_duration_min"] - before_after["after_ems_duration_min"]
    )
    before_after["hospital_improvement_min"] = (
        before_after["before_hospital_duration_min"]
        - before_after["after_hospital_duration_min"]
    )
    before_after["total_improvement_min"] = (
        before_after["ems_improvement_min"] + before_after["hospital_improvement_min"]
    )

    summary = {
        "source_method": source_method,
        "source_seed": source_seed,
        "source_model": source_model,
        "policy_budget": len(policies),
        "expected_reward": expected_reward,
        "calculated_reward": float(calculated_reward),
        "dispatch_response_improvement_min": float(dispatch_improvement),
        "selected_policies": policies,
        "baseline_metrics": baseline_metrics,
        "final_metrics": final_metrics,
    }
    return comparison, before_after, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/RL")
    parser.add_argument("--results-dir", default="results/RL/refined_with_dispatch")
    args = parser.parse_args()

    comparison, before_after, summary = build_reports(args.data_dir, args.results_dir)
    output = Path(args.results_dir)
    comparison.to_csv(output / "model_comparison.csv", index=False, encoding="utf-8-sig")
    before_after.to_csv(output / "final_before_after_by_region.csv", index=False, encoding="utf-8-sig")
    (output / "final_policy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(summary["selected_policies"]).to_csv(
        output / "final_selected_policies.csv", index=False, encoding="utf-8-sig"
    )
    print(comparison.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
