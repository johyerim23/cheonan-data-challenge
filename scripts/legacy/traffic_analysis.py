"""도로구간별 교통 혼잡도 지수 계산.

혼잡도 = normalize(평균교통량) × (1 - normalize(평균속도))
"""
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PROCESSED_DIR / "도로구간_소통통계.csv"
OUTPUT_PATH = ANALYSIS_DIR / "교통혼잡도_도로구간.csv"
SUMMARY_PATH = ANALYSIS_DIR / "교통혼잡도_시간대요약.csv"


def minmax(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    low, high = series.min(), series.max()
    if pd.isna(low) or pd.isna(high):
        return pd.Series(float("nan"), index=series.index)
    if high == low:
        return pd.Series(0.5, index=series.index)
    return (series - low) / (high - low)


def congestion_level(score: float) -> str:
    if score >= 0.6:
        return "매우 혼잡"
    if score >= 0.4:
        return "혼잡"
    if score >= 0.2:
        return "보통"
    return "원활"


def build_congestion_index() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    required = ["도로명", "구간명", "평균속도", "교통량"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"전처리 데이터 필수 컬럼이 없습니다: {missing}")

    df["평균속도_norm"] = minmax(df["평균속도"])
    df["교통량_norm"] = minmax(df["교통량"])
    df["저속도_norm"] = 1 - df["평균속도_norm"]
    df["교통혼잡도"] = df["교통량_norm"] * df["저속도_norm"]
    df["혼잡등급"] = df["교통혼잡도"].apply(congestion_level)
    df = df.sort_values("교통혼잡도", ascending=False).reset_index(drop=True)
    df.insert(0, "혼잡순위", range(1, len(df) + 1))
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    if "혼잡시간" in df.columns:
        summary = (df.dropna(subset=["혼잡시간"])
                     .groupby("혼잡시간", as_index=False)
                     .agg(구간수=("구간명", "count"),
                          평균교통량=("교통량", "mean"),
                          평균속도=("평균속도", "mean"),
                          평균혼잡도=("교통혼잡도", "mean"))
                     .sort_values("혼잡시간"))
        summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    result = build_congestion_index()
    print("교통 혼잡도 계산 완료")
    print(result[["혼잡순위", "도로명", "구간명", "평균속도", "교통량", "교통혼잡도"]].head(10))
    print(f"저장 위치: {OUTPUT_PATH}")
