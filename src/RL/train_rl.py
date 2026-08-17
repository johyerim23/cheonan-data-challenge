"""Train a DQN policy for the Cheonan emergency policy environment."""

import argparse
import json
import time
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from env import CheonanEmergencyEnv


def train_dqn(
    data_dir,
    model_path,
    total_timesteps,
    policy_budget,
    seed,
    progress_bar=False,
    check_environment=True,
    verbose=0,
):
    """Train and save one reproducible DQN run."""

    raw_env = CheonanEmergencyEnv(data_dir=data_dir, policy_budget=policy_budget)
    if check_environment:
        check_env(raw_env, warn=True)

    action_count = int(raw_env.action_space.n)
    env = Monitor(raw_env)
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=min(1_000, max(1, total_timesteps // 5)),
        batch_size=128,
        gamma=0.99,
        exploration_fraction=0.30,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        target_update_interval=500,
        train_freq=4,
        gradient_steps=1,
        verbose=verbose,
        seed=seed,
        device="cpu",
    )

    print("=" * 60)
    print("DQN 학습 시작")
    print("=" * 60)
    print(f"총 timestep: {total_timesteps}")
    print(f"정책 예산: {policy_budget}")
    print(f"Action 수: {action_count}")
    print(f"Seed: {seed}")

    started_at = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)
    elapsed_seconds = time.perf_counter() - started_at

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    metadata = {
        "model_path": str(model_path.with_suffix(".zip")),
        "data_dir": data_dir,
        "total_timesteps": total_timesteps,
        "policy_budget": policy_budget,
        "action_count": action_count,
        "seed": seed,
        "elapsed_seconds": elapsed_seconds,
        "device": "cpu",
    }
    metadata_path = model_path.with_name(model_path.name + "_metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    env.close()
    print("\n" + "=" * 60)
    print("DQN 학습 완료")
    print("=" * 60)
    print(f"저장 위치: {model_path.with_suffix('.zip')}")
    print(f"학습 시간: {elapsed_seconds:.1f}초")
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="천안 응급의료 DQN을 학습합니다.")
    parser.add_argument("--data-dir", default="data/processed/RL")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--policy-budget", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--skip-env-check", action="store_true")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.total_timesteps <= 0:
        raise ValueError("total_timesteps는 1 이상이어야 합니다.")
    if args.policy_budget <= 0:
        raise ValueError("policy_budget은 1 이상이어야 합니다.")

    model_path = args.model_path
    if model_path is None:
        model_path = (
            f"models/RL/cheonan_dqn_refined_{args.total_timesteps}_seed{args.seed}"
        )

    train_dqn(
        data_dir=args.data_dir,
        model_path=model_path,
        total_timesteps=args.total_timesteps,
        policy_budget=args.policy_budget,
        seed=args.seed,
        progress_bar=args.progress_bar,
        check_environment=not args.skip_env_check,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
