from env import CheonanEmergencyEnv


env = CheonanEmergencyEnv(
    data_dir="data/processed/RL",
    policy_budget=3
)


# =========================
# 환경 확인
# =========================

print("=" * 60)
print("환경 확인")
print("=" * 60)

print(
    "Action 수:",
    env.action_space.n
)

print(
    "State 크기:",
    env.observation_space.shape
)

print()


# =========================
# Reset
# =========================

state, info = env.reset()

print("=" * 60)
print("초기 State")
print("=" * 60)

print(
    "State shape:",
    state.shape
)

print(
    "State min:",
    state.min()
)

print(
    "State max:",
    state.max()
)

print()


# =========================
# Action 목록 확인
# =========================

print("=" * 60)
print("Action 후보")
print("=" * 60)

for i, action in enumerate(
    env.actions[:10]
):

    print(
        i,
        action
    )

print(
    "..."
)

print()


# =========================
# Random Action 테스트
# =========================

print("=" * 60)
print("Random 정책 선택 테스트")
print("=" * 60)

done = False

total_reward = 0


while not done:

    action = (
        env.action_space.sample()
    )

    state, reward, terminated, truncated, info = (
        env.step(action)
    )

    total_reward += reward

    print()
    print(
        "선택:",
        info[
            "selected_action"
        ]
    )

    print(
        "Reward:",
        round(
            reward,
            4
        )
    )

    print(
        "남은 예산:",
        info[
            "remaining_budget"
        ]
    )

    done = (
        terminated
        or truncated
    )


print()

print("=" * 60)
print("Episode 종료")
print("=" * 60)

print(
    "총 Reward:",
    round(
        total_reward,
        4
    )
)

print()

print(
    "최종 metrics:"
)

for key, value in (
    info[
        "metrics"
    ].items()
):

    print(
        key,
        ":",
        round(
            value,
            3
        )
    )