"""Create a comparable Random vs Rule-based vs DQN evaluation table."""

import argparse
import json
from pathlib import Path

import pandas as pd
from stable_baselines3 import DQN

from env import CheonanEmergencyEnv


METRIC_COLUMNS = [
    "mean_ems_time",
    "mean_hospital_time",
    "mean_total_time",
    "weighted_ems_time",
    "weighted_hospital_time",
    "weighted_total_time",
    "max_ems_time",
    "max_hospital_time",
]


def evaluate_dqn(model_path, data_dir, policy_budget):
    env = CheonanEmergencyEnv(
        data_dir=data_dir,
        policy_budget=policy_budget,
        refined_ambulance_actions=False,
    )
    model = DQN.load(model_path, env=env, device="cpu")
    state, _ = env.reset()
    total_reward = 0.0
    selected_policies = []
    info = None
    done = False

    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        selected_policies.append(info["selected_action"])
        done = terminated or truncated

    if info is None:
        raise RuntimeError("DQN 평가에서 실행된 Action이 없습니다.")

    env.close()
    return total_reward, info["metrics"], selected_policies


def build_comparison(
    results_dir,
    model_path,
    data_dir,
    policy_budget,
    dqn_seed_results=None,
):
    results_dir = Path(results_dir)
    random_episodes = pd.read_csv(results_dir / "random_episodes.csv")
    rule_summary = pd.read_csv(results_dir / "rule_summary.csv").iloc[0]

    dqn_reward, dqn_metrics, dqn_policies = evaluate_dqn(
        model_path=model_path,
        data_dir=data_dir,
        policy_budget=policy_budget,
    )

    random_row = {
        "method": "random",
        "evaluation_units": len(random_episodes),
        "reward_mean": random_episodes["reward"].mean(),
        "reward_std": random_episodes["reward"].std(ddof=1),
        "reward_min": random_episodes["reward"].min(),
        "reward_max": random_episodes["reward"].max(),
        **{
            column: random_episodes[column].mean()
            for column in METRIC_COLUMNS
        },
        "selected_policies": "",
    }

    rule_row = {
        "method": "rule_greedy",
        "evaluation_units": 1,
        "reward_mean": rule_summary["reward"],
        "reward_std": 0.0,
        "reward_min": rule_summary["reward"],
        "reward_max": rule_summary["reward"],
        **{column: rule_summary[column] for column in METRIC_COLUMNS},
        "selected_policies": rule_summary["selected_policies"],
    }

    dqn_row = {
        "method": "dqn_5000",
        "evaluation_units": 1,
        "reward_mean": dqn_reward,
        "reward_std": 0.0,
        "reward_min": dqn_reward,
        "reward_max": dqn_reward,
        **dqn_metrics,
        "selected_policies": json.dumps(dqn_policies, ensure_ascii=False),
    }

    rows = [random_row, rule_row, dqn_row]

    if dqn_seed_results is not None and Path(dqn_seed_results).exists():
        seed_results = pd.read_csv(dqn_seed_results)
        rows.append(
            {
                "method": "dqn_50000_multi_seed",
                "evaluation_units": len(seed_results),
                "reward_mean": seed_results["reward"].mean(),
                "reward_std": seed_results["reward"].std(ddof=1),
                "reward_min": seed_results["reward"].min(),
                "reward_max": seed_results["reward"].max(),
                **{
                    column: seed_results[column].mean()
                    for column in METRIC_COLUMNS
                },
                "selected_policies": "seed별 결과는 dqn_seed_evaluation.csv 참조",
            }
        )

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Random/Rule/DQN 결과를 비교합니다.")
    parser.add_argument("--results-dir", default="results/RL")
    parser.add_argument("--model-path", default="models/RL/cheonan_dqn")
    parser.add_argument("--data-dir", default="data/processed/RL")
    parser.add_argument("--policy-budget", type=int, default=3)
    parser.add_argument(
        "--dqn-seed-results",
        default="results/RL/dqn_seed_evaluation.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    comparison = build_comparison(
        results_dir=args.results_dir,
        model_path=args.model_path,
        data_dir=args.data_dir,
        policy_budget=args.policy_budget,
        dqn_seed_results=args.dqn_seed_results,
    )

    output_path = Path(args.results_dir) / "model_comparison.csv"
    comparison.to_csv(output_path, index=False, encoding="utf-8-sig")

    display_columns = [
        "method",
        "evaluation_units",
        "reward_mean",
        "reward_std",
        "weighted_total_time",
        "mean_total_time",
        "max_ems_time",
        "max_hospital_time",
    ]
    print(comparison[display_columns].to_string(index=False))
    print(f"\n비교 결과: {output_path}")


if __name__ == "__main__":
    main()
