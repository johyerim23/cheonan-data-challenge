"""Evaluate a deterministic greedy rule-based policy baseline."""

import argparse
import json
from pathlib import Path

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
DEFAULT_POLICY_BUDGET = 3


def evaluate_candidate(sim, policy, before_weighted_total, increment_effects):
    """Return immediate reward without changing the simulator permanently."""

    saved = sim.snapshot()

    apply_policy(sim, policy)
    after_weighted_total = sim.calculate_metrics()["weighted_total_time"]

    sim.restore(saved)

    return (
        before_weighted_total - after_weighted_total
        + policy_reward_bonus(policy, increment_effects)
    )


def evaluate_rule(
    data_dir=DEFAULT_DATA_DIR,
    policy_budget=DEFAULT_POLICY_BUDGET,
    refined_ambulance_actions=True,
):
    """Select the action with the greatest immediate reward at each step."""

    sim = EmergencyPolicySimulator(data_dir=data_dir)
    actions = build_policy_actions(
        sim,
        refined_ambulance_actions=refined_ambulance_actions,
    )

    if policy_budget <= 0:
        raise ValueError("policy_budget은 1 이상이어야 합니다.")
    if policy_budget > len(actions):
        raise ValueError(
            f"policy_budget({policy_budget})가 Action 수({len(actions)})보다 큽니다."
        )

    baseline_metrics = sim.calculate_metrics()
    increment_effects = load_ambulance_increment_effects(data_dir)
    unused_indices = set(range(len(actions)))
    selections = []
    total_reward = 0.0
    cumulative_dispatch_improvement = 0.0

    for step in range(1, policy_budget + 1):
        before = sim.calculate_metrics()
        before_weighted_total = before["weighted_total_time"]
        candidate_rewards = []

        for action_index in sorted(unused_indices):
            reward = evaluate_candidate(
                sim,
                actions[action_index],
                before_weighted_total,
                increment_effects,
            )
            candidate_rewards.append((reward, action_index))

        # The smaller action index is the deterministic tie breaker.
        best_reward, best_index = max(
            candidate_rewards,
            key=lambda item: (item[0], -item[1]),
        )
        best_policy = actions[best_index]
        apply_policy(sim, best_policy)
        unused_indices.remove(best_index)
        total_reward += best_reward
        dispatch_improvement = policy_reward_bonus(best_policy, increment_effects)
        cumulative_dispatch_improvement += dispatch_improvement

        after = sim.calculate_metrics()
        selections.append(
            {
                "step": step,
                "action_index": best_index,
                **best_policy,
                "step_reward": best_reward,
                "cumulative_reward": total_reward,
                "dispatch_response_improvement_min": dispatch_improvement,
                "weighted_total_time_before": before_weighted_total,
                "weighted_total_time_after": after["weighted_total_time"],
            }
        )

    final_metrics = sim.calculate_metrics()
    summary = pd.DataFrame(
        [
            {
                "method": "rule_greedy_immediate_reward",
                "policy_budget": policy_budget,
                "action_count": len(actions),
                "reward": total_reward,
                "baseline_weighted_total_time": baseline_metrics[
                    "weighted_total_time"
                ],
                **final_metrics,
                "dispatch_response_improvement_min": cumulative_dispatch_improvement,
                "selected_action_indices": json.dumps(
                    [row["action_index"] for row in selections],
                    ensure_ascii=False,
                ),
                "selected_policies": json.dumps(
                    [
                        {
                            "policy": row["policy"],
                            "target_id": row["target_id"],
                            "target_name": row["target_name"],
                        }
                        for row in selections
                    ],
                    ensure_ascii=False,
                ),
            }
        ]
    )

    return pd.DataFrame(selections), summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="즉시 Reward 최대화 규칙 기반 정책을 평가합니다."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policy-budget", type=int, default=DEFAULT_POLICY_BUDGET)
    parser.add_argument("--legacy-actions", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    selections, summary = evaluate_rule(
        data_dir=args.data_dir,
        policy_budget=args.policy_budget,
        refined_ambulance_actions=not args.legacy_actions,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selections_path = output_dir / "rule_selections.csv"
    summary_path = output_dir / "rule_summary.csv"
    selections.to_csv(selections_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("Rule-based baseline 평가 완료")
    print("=" * 60)
    print(selections.to_string(index=False))
    print("\n", summary.to_string(index=False))
    print(f"\n선택 결과: {selections_path}")
    print(f"요약 결과: {summary_path}")


if __name__ == "__main__":
    main()
