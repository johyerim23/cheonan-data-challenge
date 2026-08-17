import pandas as pd
import os

# =========================
# 1. 경로 설정
# =========================
DATA_DIR = "data/processed/최종파일_여원"
OUT_DIR = "data/processed/RL"
OUT_PATH = os.path.join(OUT_DIR, "region_features.csv")

# =========================
# 2. 최종 데이터 불러오기
# =========================
ems = pd.read_csv(
    os.path.join(DATA_DIR, "final_regional_ems_travel_time.csv")
)

hospital = pd.read_csv(
    os.path.join(DATA_DIR, "final_regional_hospital_travel_time.csv")
)

alternative = pd.read_csv(
    os.path.join(DATA_DIR, "final_regional_alternative_hospital.csv")
)

traffic = pd.read_csv(
    os.path.join(DATA_DIR, "final_traffic_congestion.csv")
)

vulnerable = pd.read_csv(
    os.path.join(DATA_DIR, "final_regional_vulnerable_population.csv")
)

occurrence = pd.read_csv(
    os.path.join(DATA_DIR, "final_regional_emergency_occurrence_risk.csv")
)


# =========================
# 3. 지역명 컬럼 통일
# =========================
traffic = traffic.rename(columns={"행정동": "region_name"})
vulnerable = vulnerable.rename(columns={"행정동": "region_name"})


# =========================
# 4. 필요한 컬럼만 선택
# =========================
ems = ems[
    [
        "region_id",
        "region_name",
        "station_id",
        "station_name",
        "ems_distance_km",
        "ems_duration_min",
        "ems_travel_time_score",
    ]
]

hospital = hospital[
    [
        "region_name",
        "hospital_id",
        "hospital_name",
        "hospital_type",
        "hospital_distance_km",
        "hospital_duration_min",
        "hospital_travel_time_score",
    ]
]

alternative = alternative[
    [
        "region_name",
        "alternative_hospital_count",
        "alternative_supply_score",
        "nearest_alternative_hospital_id",
        "nearest_alternative_hospital_name",
        "nearest_alternative_hospital_type",
        "nearest_alternative_time_min",
        "alternative_hospital_score",
    ]
]

traffic = traffic[
    [
        "region_name",
        "교통혼잡도_v2",
    ]
]

vulnerable = vulnerable[
    [
        "region_name",
        "vulnerable_population_score",
    ]
]

occurrence = occurrence[
    [
        "region_name",
        "emergency_calls",
        "calls_per_10000",
        "emergency_occurrence_risk_score",
    ]
]


# =========================
# 5. 31개 읍면동 기준 통합
# =========================
df = (
    ems
    .merge(hospital, on="region_name", how="inner")
    .merge(alternative, on="region_name", how="inner")
    .merge(traffic, on="region_name", how="inner")
    .merge(vulnerable, on="region_name", how="inner")
    .merge(occurrence, on="region_name", how="inner")
)


# =========================
# 6. RL 입력용 0~1 값 생성
# =========================
df["ems_score_01"] = df["ems_travel_time_score"] / 100
df["hospital_score_01"] = df["hospital_travel_time_score"] / 100
df["alternative_score_01"] = df["alternative_hospital_score"] / 100
df["traffic_score_01"] = df["교통혼잡도_v2"] / 100
df["vulnerable_score_01"] = df["vulnerable_population_score"] / 100
df["occurrence_score_01"] = df["emergency_occurrence_risk_score"] / 100


# =========================
# 7. 대체 취약도용 ETA 차이
# =========================
# 2순위 ETA - 1순위 ETA
df["alternative_gap_min"] = (
    df["nearest_alternative_time_min"]
    - df["hospital_duration_min"]
).clip(lower=0)


# =========================
# 8. 5개 취약유형 분류
# =========================

TYPE_QUANTILE = 0.70   # 상위 30%

DISPATCH_CUTOFF = df["ems_duration_min"].quantile(TYPE_QUANTILE)
TRANSFER_CUTOFF = df["hospital_duration_min"].quantile(TYPE_QUANTILE)
ALTERNATIVE_CUTOFF = df["alternative_gap_min"].quantile(TYPE_QUANTILE)
TRAFFIC_CUTOFF = df["traffic_score_01"].quantile(TYPE_QUANTILE)
DEMAND_CUTOFF = df["vulnerable_score_01"].quantile(TYPE_QUANTILE)

print("출동 cutoff:", round(DISPATCH_CUTOFF, 2))
print("이송 cutoff:", round(TRANSFER_CUTOFF, 2))
print("대체 cutoff:", round(ALTERNATIVE_CUTOFF, 2))
print("교통 cutoff:", round(TRAFFIC_CUTOFF, 2))
print("수요 cutoff:", round(DEMAND_CUTOFF, 2))

df["type_dispatch"] = (
    df["ems_duration_min"] >= DISPATCH_CUTOFF
)

df["type_transfer"] = (
    df["hospital_duration_min"] >= TRANSFER_CUTOFF
)

df["type_alternative"] = (
    df["alternative_gap_min"] >= ALTERNATIVE_CUTOFF
)

df["type_traffic"] = (
    df["traffic_score_01"] >= TRAFFIC_CUTOFF
)

df["type_demand"] = (
    df["vulnerable_score_01"] >= DEMAND_CUTOFF
)


# =========================
# 9. 유형 개수 및 라벨 생성
# =========================
type_cols = [
    "type_dispatch",
    "type_transfer",
    "type_alternative",
    "type_traffic",
    "type_demand",
]

df["vulnerability_type_count"] = (
    df[type_cols].sum(axis=1)
)


def make_type_label(row):
    labels = []

    if row["type_dispatch"]:
        labels.append("출동")
    if row["type_transfer"]:
        labels.append("이송")
    if row["type_alternative"]:
        labels.append("대체")
    if row["type_traffic"]:
        labels.append("교통")
    if row["type_demand"]:
        labels.append("수요")

    if len(labels) == 0:
        return "상대적 비취약"

    return "+".join(labels)


df["vulnerability_types"] = df.apply(
    make_type_label,
    axis=1
)

df["is_complex_type"] = (
    df["vulnerability_type_count"] >= 2
)


# =========================
# 10. 검증
# =========================
print("지역 수:", len(df))
print("고유 지역 수:", df["region_name"].nunique())

assert len(df) == 31
assert df["region_name"].nunique() == 31

check_cols = [
    "ems_score_01",
    "hospital_score_01",
    "alternative_score_01",
    "traffic_score_01",
    "vulnerable_score_01",
    "occurrence_score_01",
]

print("\n결측치")
print(df[check_cols].isna().sum())

print("\n취약유형 개수")
print("출동:", df["type_dispatch"].sum())
print("이송:", df["type_transfer"].sum())
print("대체:", df["type_alternative"].sum())
print("교통:", df["type_traffic"].sum())
print("수요:", df["type_demand"].sum())


# =========================
# 11. 저장
# =========================
df = df.sort_values("region_id").reset_index(drop=True)

df.to_csv(
    OUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\n저장 완료:")
print(OUT_PATH)