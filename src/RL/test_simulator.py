from simulator import EmergencyPolicySimulator


DATA_DIR = "data/processed/RL"


sim = EmergencyPolicySimulator(
    data_dir=DATA_DIR
)


# =========================
# 1. 데이터 확인
# =========================
print("=" * 60)
print("데이터 확인")
print("=" * 60)

print(
    "지역 수:",
    len(sim.state)
)

print(
    "EMS 거점 수:",
    len(sim.current_ems)
)

print(
    "병원 업그레이드 후보 수:",
    len(sim.hospital_candidates)
)

print()


# =========================
# 2. 데이터 기반 보정값 확인
# =========================
print("=" * 60)
print("이동시간 보정값")
print("=" * 60)

print(
    "EMS 직선→도로거리 계수:",
    round(
        sim.ems_circuity,
        3
    )
)

print(
    "EMS 분/km:",
    round(
        sim.ems_min_per_km,
        3
    )
)

print(
    "병원 직선→도로거리 계수:",
    round(
        sim.hospital_circuity,
        3
    )
)

print(
    "병원 분/km:",
    round(
        sim.hospital_min_per_km,
        3
    )
)

print()


# =========================
# 3. Before
# =========================
print("=" * 60)
print("Before")
print("=" * 60)

before = sim.calculate_metrics()

for key, value in before.items():
    print(
        f"{key}: "
        f"{value:.3f}"
    )


# =========================
# 4. 신규 구급거점 정책 테스트
# =========================
target_region = sim.state.loc[
    sim.state[
        "ems_duration_min"
    ].idxmax(),
    "region_id"
]

target_name = sim.state.loc[
    sim.state[
        "region_id"
    ] == target_region,
    "region_name"
].iloc[0]

print()
print("=" * 60)
print("신규 구급거점 정책 테스트")
print("=" * 60)

print(
    "대상 지역:",
    target_name,
    target_region
)

sim.add_new_ambulance_base(
    target_region
)


# =========================
# 5. After
# =========================
after = sim.calculate_metrics()

print()

for key, value in after.items():
    print(
        f"{key}: "
        f"{value:.3f}"
    )


# =========================
# 6. Before / After
# =========================
print()
print("=" * 60)
print("Before / After")
print("=" * 60)

comparison = (
    sim.compare_with_baseline()
)

for key, values in comparison.items():

    print(
        f"{key}: "
        f"{values['before']:.3f}"
        f" -> "
        f"{values['after']:.3f}"
        f" | 개선 "
        f"{values['improvement']:.3f}"
    )


print()
print(
    "Reward:",
    round(
        sim.calculate_reward(),
        3
    )
)


# =========================
# 7. Reset
# =========================
sim.reset()


# =========================
# 8. 병원 업그레이드 테스트
# =========================
print()
print("=" * 60)
print("병원 업그레이드 테스트")
print("=" * 60)

print(
    sim.hospital_candidates[
        [
            "institution_id",
            "name",
            "address"
        ]
    ].head(10)
)


candidate_id = (
    sim.hospital_candidates
    .iloc[0][
        "institution_id"
    ]
)

candidate_name = (
    sim.hospital_candidates
    .iloc[0]["name"]
)

print()
print(
    "테스트 병원:",
    candidate_name,
    candidate_id
)

sim.upgrade_hospital(
    candidate_id
)


comparison = (
    sim.compare_with_baseline()
)

for key, values in comparison.items():

    print(
        f"{key}: "
        f"{values['before']:.3f}"
        f" -> "
        f"{values['after']:.3f}"
        f" | 개선 "
        f"{values['improvement']:.3f}"
    )


print()
print(
    "Reward:",
    round(
        sim.calculate_reward(),
        3
    )
)


# =========================
# 9. 기존센터 증차 테스트
# =========================
sim.reset()

print()
print("=" * 60)
print("기존 센터 증차 테스트")
print("=" * 60)

station_id = (
    sim.current_ems
    .iloc[0]["station_id"]
)

before_count = (
    sim.current_ems.loc[
        sim.current_ems[
            "station_id"
        ] == station_id,
        "ambulance_count"
    ].iloc[0]
)

sim.add_ambulance_existing_center(
    station_id
)

after_count = (
    sim.current_ems.loc[
        sim.current_ems[
            "station_id"
        ] == station_id,
        "ambulance_count"
    ].iloc[0]
)

print(
    station_id,
    ":",
    before_count,
    "->",
    after_count
)


print()
print("=" * 60)
print("테스트 완료")
print("=" * 60)