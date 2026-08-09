"""교통 혼잡도 전처리.

원본 ``data/raw/소통통계.xlsx``에서 도로구간별 평균속도와 교통량을
정리하여 ``data/processed/도로구간_소통통계.csv``로 저장한다.
"""
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = PROCESSED_DIR / "도로구간_소통통계.csv"


def find_traffic_file() -> Path:
    """파일명이 달라져도 '소통통계' 시트/필수 컬럼을 기준으로 원본을 찾는다."""
    preferred = RAW_DIR / "소통통계.xlsx"
    if preferred.exists():
        return preferred

    for path in RAW_DIR.glob("*.xlsx"):
        try:
            xls = pd.ExcelFile(path)
            if "소통통계" in xls.sheet_names:
                return path
            sample = pd.read_excel(path, nrows=1)
            cols = {str(c).strip() for c in sample.columns}
            if {"도로명", "구간명", "평균속도(km/h)", "교통량(대)"}.issubset(cols):
                return path
        except Exception:
            continue
    raise FileNotFoundError(f"{RAW_DIR}에서 소통통계 엑셀 파일을 찾지 못했습니다.")


def process_traffic(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path) if path else find_traffic_file()
    sheet = "소통통계" if "소통통계" in pd.ExcelFile(path).sheet_names else 0
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={
        "평균속도(km/h)": "평균속도",
        "교통량(대)": "교통량",
    })

    required = ["도로명", "구간명", "평균속도", "교통량"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"소통통계 필수 컬럼이 없습니다: {missing}")

    keep = required + [c for c in ["통행속도 최저(km/h)", "통행속도 최고(km/h)", "혼잡시간"] if c in df.columns]
    df = df[keep].copy()
    df["도로명"] = df["도로명"].astype(str).str.strip()
    df["구간명"] = df["구간명"].astype(str).str.strip()
    for col in ["평균속도", "교통량", "혼잡시간"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).copy()
    df = df[(df["평균속도"] > 0) & (df["교통량"] >= 0)]
    df = df.drop_duplicates(subset=["도로명", "구간명"], keep="last")
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    result = process_traffic()
    print(f"교통 데이터 전처리 완료: {len(result):,}개 도로구간")
    print(f"저장 위치: {OUTPUT_PATH}")
    print(result.head())
