"""Evaluate one or more trained DQN models deterministically."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from stable_baselines3 import DQN

from env import CheonanEmergencyEnv


def evaluate_model(model_path, data_dir, policy_budget, refined_ambulance_actions=False):
    env = CheonanEmergencyEnv(
        data_dir=data_dir,
        policy_budget=policy_budget,
        refined_ambulance_actions=refined_ambulance_actions,
    )
    model = DQN.load(str(model_path), env=env, device="cpu")
    state, _ = env.reset()
    total_reward = 0.0
    selected_policies = []
    selected_indices = []
    step_rewards = []
    info = None
    done = False

    while not done:
        action, _ = model.predict(state, deterministic=True)
        action_index = int(action)
        state, reward, terminated, truncated, info = env.step(action_index)
        selected_indices.append(action_index)
        selected_policies.append(info["selected_action"])
        step_rewards.append(float(reward))
        total_reward += reward
        done = terminated or truncated

    if info is None:
        raise RuntimeError("DQN 평가에서 실행된 Action이 없습니다.")

    model_name = Path(model_path).stem
    seed_match = re.search(r"seed(\d+)", model_name)
    timestep_match = re.search(r"dqn_(?:refined_)?(\d+)_seed", model_name)
    result = {
        "model_name": model_name,
        "model_path": str(model_path),
        "seed": int(seed_match.group(1)) if seed_match else None,
        "total_timesteps": (
            int(timestep_match.group(1)) if timestep_match else None
        ),
        "reward": float(total_reward),
        **info["metrics"],
        "has_repeated_action": len(set(selected_indices)) != len(selected_indices),
        "selected_action_indices": json.dumps(selected_indices),
        "step_rewards": json.dumps(step_rewards),
        "selected_policies": json.dumps(selected_policies, ensure_ascii=False),
    }
    env.close()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="학습된 DQN 모델을 평가합니다.")
    parser.add_argument("model_paths", nargs="+")
    parser.add_argument("--data-dir", default="data/processed/RL")
    parser.add_argument("--policy-budget", type=int, default=3)
    parser.add_argument("--output", default="results/RL/dqn_seed_evaluation.csv")
    parser.add_argument("--refined-actions", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [
        evaluate_model(
            path,
            args.data_dir,
            args.policy_budget,
            refined_ambulance_actions=args.refined_actions,
        )
        for path in args.model_paths
    ]
    results = pd.DataFrame(rows).sort_values(
        ["total_timesteps", "seed"], na_position="first"
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {
                "method": "dqn",
                "model_count": len(results),
                "total_timesteps": results["total_timesteps"].iloc[0],
                "reward_mean": results["reward"].mean(),
                "reward_std": results["reward"].std(ddof=1),
                "reward_min": results["reward"].min(),
                "reward_max": results["reward"].max(),
                "weighted_total_time_mean": results[
                    "weighted_total_time"
                ].mean(),
                "weighted_total_time_std": results[
                    "weighted_total_time"
                ].std(ddof=1),
                "mean_total_time_mean": results["mean_total_time"].mean(),
                "repeated_action_model_count": int(
                    results["has_repeated_action"].sum()
                ),
                "unique_policy_combinations": results[
                    "selected_action_indices"
                ].nunique(),
            }
        ]
    )
    summary_path = output_path.with_name("dqn_seed_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    columns = [
        "model_name",
        "seed",
        "total_timesteps",
        "reward",
        "weighted_total_time",
        "mean_total_time",
        "has_repeated_action",
    ]
    print(results[columns].to_string(index=False))
    print("\n", summary.to_string(index=False))
    print(f"\n평가 결과: {output_path}")
    print(f"요약 결과: {summary_path}")


if __name__ == "__main__":
    main()
