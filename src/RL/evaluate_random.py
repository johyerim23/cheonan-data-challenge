"""Evaluate a uniform random policy baseline for the Cheonan RL environment."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulator import EmergencyPolicySimulator
from policy_actions import (
    apply_policy,
    build_policy_actions,
    load_ambulance_increment_effects,
    policy_reward_bonus,
)


DEFAULT_DATA_DIR = "data/processed/RL"
DEFAULT_OUTPUT_DIR = "results/RL"
DEFAULT_EPISODES = 1_000
DEFAULT_POLICY_BUDGET = 3
DEFAULT_SEED = 42


def evaluate_random(
    data_dir=DEFAULT_DATA_DIR,
    episodes=DEFAULT_EPISODES,
    policy_budget=DEFAULT_POLICY_BUDGET,
    seed=DEFAULT_SEED,
    refined_ambulance_actions=True,
):
    """Run episodes using uniformly sampled, non-repeating actions."""

    sim = EmergencyPolicySimulator(data_dir=data_dir)
    actions = build_policy_actions(
        sim,
        refined_ambulance_actions=refined_ambulance_actions,
    )

    if policy_budget > len(actions):
        raise ValueError(
            f"policy_budget({policy_budget})가 Action 수({len(actions)})보다 큽니다."
        )

    rng = np.random.default_rng(seed)
    baseline_metrics = sim.calculate_metrics()
    increment_effects = load_ambulance_increment_effects(data_dir)
    rows = []

    for episode in range(episodes):
        sim.reset()
        sampled_actions = rng.choice(
            len(actions),
            size=policy_budget,
            replace=False,
        )

        total_reward = 0.0
        cumulative_dispatch_improvement = 0.0
        selected_policies = []
        final_metrics = None

        for action in sampled_actions:
            before = sim.calculate_metrics()
            policy = actions[int(action)]

            apply_policy(sim, policy)

            final_metrics = sim.calculate_metrics()
            dispatch_improvement = policy_reward_bonus(policy, increment_effects)
            cumulative_dispatch_improvement += dispatch_improvement
            total_reward += (
                before["weighted_total_time"]
                - final_metrics["weighted_total_time"]
                + dispatch_improvement
            )
            selected_policies.append(policy)

        if final_metrics is None:
            raise RuntimeError("에피소드에서 실행된 Action이 없습니다.")

        row = {
            "episode": episode + 1,
            "seed": seed,
            "reward": float(total_reward),
            "dispatch_response_improvement_min": cumulative_dispatch_improvement,
            **final_metrics,
            "selected_action_indices": json.dumps(
                [int(action) for action in sampled_actions],
                ensure_ascii=False,
            ),
            "selected_policies": json.dumps(
                selected_policies,
                ensure_ascii=False,
            ),
        }
        rows.append(row)

    results = pd.DataFrame(rows)
    reward = results["reward"]

    summary = pd.DataFrame(
        [
            {
                "method": "random",
                "episodes": episodes,
                "policy_budget": policy_budget,
                "seed": seed,
                "action_count": len(actions),
                "reward_mean": reward.mean(),
                "reward_std": reward.std(ddof=1),
                "reward_min": reward.min(),
                "reward_q25": reward.quantile(0.25),
                "reward_median": reward.median(),
                "reward_q75": reward.quantile(0.75),
                "reward_max": reward.max(),
                "baseline_weighted_total_time": baseline_metrics[
                    "weighted_total_time"
                ],
                "weighted_total_time_mean": results[
                    "weighted_total_time"
                ].mean(),
                "weighted_total_time_std": results[
                    "weighted_total_time"
                ].std(ddof=1),
                "dispatch_response_improvement_mean": results[
                    "dispatch_response_improvement_min"
                ].mean(),
            }
        ]
    )

    return results, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Random 정책을 여러 Episode 평가합니다."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--policy-budget", type=int, default=DEFAULT_POLICY_BUDGET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--legacy-actions", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.episodes <= 0:
        raise ValueError("episodes는 1 이상이어야 합니다.")
    if args.policy_budget <= 0:
        raise ValueError("policy_budget은 1 이상이어야 합니다.")

    results, summary = evaluate_random(
        data_dir=args.data_dir,
        episodes=args.episodes,
        policy_budget=args.policy_budget,
        seed=args.seed,
        refined_ambulance_actions=not args.legacy_actions,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "random_episodes.csv"
    summary_path = output_dir / "random_summary.csv"
    results.to_csv(episodes_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("Random baseline 평가 완료")
    print("=" * 60)
    print(summary.to_string(index=False))
    print(f"\nEpisode 결과: {episodes_path}")
    print(f"요약 결과: {summary_path}")


if __name__ == "__main__":
    main()
