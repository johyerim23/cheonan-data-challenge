"""Export real DQN Q-values and step-by-step policy traces for the dashboard."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RL_SOURCE = ROOT / "src" / "RL"
if str(RL_SOURCE) not in sys.path:
    sys.path.insert(0, str(RL_SOURCE))

from env import CheonanEmergencyEnv  # noqa: E402


DEFAULT_MODELS = [
    ROOT / "models" / "RL" / "cheonan_dqn_refined_50000_seed42.zip",
    ROOT / "models" / "RL" / "cheonan_dqn_refined_50000_seed123.zip",
    ROOT / "models" / "RL" / "cheonan_dqn_refined_50000_seed2026.zip",
]


def policy_label(policy: str) -> str:
    return {
        "ambulance_existing": "기존 거점 증차",
        "ambulance_new_base": "신규 구급 거점",
        "hospital_upgrade": "응급기능 승급",
    }.get(policy, policy)


def export_trace(model_paths: list[Path], output: Path) -> Path:
    traces = []
    for model_path in model_paths:
        env = CheonanEmergencyEnv(
            data_dir=str(ROOT / "data" / "processed" / "RL"),
            policy_budget=3,
            refined_ambulance_actions=True,
        )
        model = DQN.load(str(model_path), env=env, device="cpu")
        state, _ = env.reset()
        baseline = env.sim.calculate_metrics()
        steps = []
        cumulative_reward = 0.0

        for step_number in range(1, env.policy_budget + 1):
            observation, _ = model.policy.obs_to_tensor(state)
            with torch.no_grad():
                q_values = model.q_net(observation).cpu().numpy()[0]

            action_index = int(np.argmax(q_values))
            predicted_index = int(model.predict(state, deterministic=True)[0])
            if action_index != predicted_index:
                raise RuntimeError(
                    f"Q argmax와 model.predict 불일치: {action_index} != {predicted_index}"
                )

            ranking = np.argsort(q_values)[::-1]
            ranked_actions = []
            for rank, index in enumerate(ranking, start=1):
                action = env.actions[int(index)]
                ranked_actions.append(
                    {
                        "rank": rank,
                        "action_index": int(index),
                        "q_value": float(q_values[index]),
                        "policy": action["policy"],
                        "policy_label": policy_label(action["policy"]),
                        "target_name": action["target_name"],
                        "used_before_step": bool(env.used_actions[index]),
                        "selected": int(index) == action_index,
                    }
                )

            state, reward, terminated, truncated, info = env.step(action_index)
            cumulative_reward += float(reward)
            steps.append(
                {
                    "step": step_number,
                    "remaining_budget_before": env.policy_budget - step_number + 1,
                    "selected_action_index": action_index,
                    "selected_policy": info["selected_action"],
                    "selected_policy_label": policy_label(info["selected_action"]["policy"]),
                    "selected_q_value": float(q_values[action_index]),
                    "q_gap_to_second": float(q_values[ranking[0]] - q_values[ranking[1]]),
                    "reward": float(reward),
                    "cumulative_reward": cumulative_reward,
                    "reward_components": info["reward_components"],
                    "metrics": info["metrics"],
                    "ranked_actions": ranked_actions,
                }
            )
            if terminated or truncated:
                break

        seed_match = re.search(r"seed(\d+)", model_path.stem)
        traces.append(
            {
                "model_name": model_path.stem,
                "seed": int(seed_match.group(1)) if seed_match else None,
                "baseline_metrics": baseline,
                "steps": steps,
                "final_reward": cumulative_reward,
                "final_metrics": steps[-1]["metrics"],
            }
        )
        env.close()

    payload = {
        "schema_version": 1,
        "description": "Stable-Baselines3 DQN q_net deterministic evaluation trace",
        "state_dim": 396,
        "action_dim": 54,
        "policy_budget": 3,
        "traces": traces,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="대시보드용 DQN Q-value 추적 데이터를 생성합니다.")
    parser.add_argument("model_paths", nargs="*", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--output", type=Path, default=HERE / "dqn_trace.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(export_trace(args.model_paths or DEFAULT_MODELS, args.output))
