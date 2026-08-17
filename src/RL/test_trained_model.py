from stable_baselines3 import DQN
from env import CheonanEmergencyEnv

DATA_DIR = "data/processed/RL"
MODEL_PATH = "models/RL/cheonan_dqn"

env = CheonanEmergencyEnv(
    data_dir=DATA_DIR,
    policy_budget=3,
    refined_ambulance_actions=False,
)

model = DQN.load(
    MODEL_PATH,
    env=env,
    device="cpu"
)

state, info = env.reset()

done = False
total_reward = 0
selected_policies = []

while not done:

    action, _ = model.predict(
        state,
        deterministic=True
    )

    state, reward, terminated, truncated, info = env.step(action)

    selected_policies.append(
        info["selected_action"]
    )

    total_reward += reward

    done = terminated or truncated


print("=" * 60)
print("DQN 추천 정책 조합")
print("=" * 60)

for i, policy in enumerate(
    selected_policies,
    start=1
):
    print(f"{i}. {policy}")

print()

print(
    "총 Reward:",
    round(total_reward, 4)
)

print()

print("=" * 60)
print("최종 성능")
print("=" * 60)

for key, value in info["metrics"].items():
    print(
        key,
        ":",
        round(value, 3)
    )
