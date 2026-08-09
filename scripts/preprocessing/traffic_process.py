from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


# ============================================================
# 경로 설정
# ============================================================

ROOT_DIR = Path(
    __file__
).resolve().parents[2]

RAW_DIR = (
    ROOT_DIR
    / "data"
    / "raw"
)

PROCESSED_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ROAD_OUTPUT = (
    PROCESSED_DIR
    / "교통_도로구간_소통통계.csv"
)

SMART_HOURLY_OUTPUT = (
    PROCESSED_DIR
    / "교통_스마트교차로_시간대별.csv"
)

SMART_DAILY_OUTPUT = (
    PROCESSED_DIR
    / "교통_스마트교차로_일별요약.csv"
)

ILLEGAL_PARKING_OUTPUT = (
    PROCESSED_DIR
    / "교통_불법주정차_기초.csv"
)


# ============================================================
# 공통 함수
# ============================================================

def clean_text(
    series: pd.Series,
) -> pd.Series:

    return (
        series
        .astype("string")
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )


def find_excel_by_sheet(
    sheet_name: str,
) -> Path:

    matches = []


    for path in RAW_DIR.rglob(
        "*.xlsx"
    ):

        try:

            xls = pd.ExcelFile(
                path
            )

            if (
                sheet_name
                in xls.sheet_names
            ):

                matches.append(
                    path
                )

        except Exception:

            continue


    if not matches:

        raise FileNotFoundError(
            f"'{sheet_name}' 시트를 가진 "
            "엑셀 파일을 찾지 못했습니다."
        )


    if len(matches) > 1:

        print(
            f"[경고] {sheet_name} "
            "시트를 가진 파일이 여러 개입니다."
        )

        for path in matches:

            print(
                f"  - {path.name}"
            )


    return matches[0]


# ============================================================
# 1. 도로 소통통계
# ============================================================

def process_road_traffic(
    path: Path | str | None = None,
) -> pd.DataFrame:

    path = (
        Path(path)
        if path
        else find_excel_by_sheet(
            "소통통계"
        )
    )


    df = pd.read_excel(
        path,
        sheet_name="소통통계",
    )


    df.columns = [
        str(col).strip()
        for col in df.columns
    ]


    rename_map = {

        "평균속도(km/h)":
            "평균속도_kmh",

        "통행속도 최저(km/h)":
            "최저속도_kmh",

        "통행속도 최고(km/h)":
            "최고속도_kmh",

        "교통량(대)":
            "교통량_대",

        "혼잡시간":
            "대표혼잡시간",
    }


    df = df.rename(
        columns=rename_map
    )


    required = [
        "도로명",
        "구간명",
        "평균속도_kmh",
        "교통량_대",
    ]


    missing = [
        col
        for col in required
        if col not in df.columns
    ]


    if missing:

        raise ValueError(
            "소통통계 필수 컬럼이 없습니다: "
            f"{missing}"
        )


    optional = [
        "최저속도_kmh",
        "최고속도_kmh",
        "대표혼잡시간",
    ]


    keep = (
        required
        + [
            col
            for col in optional
            if col in df.columns
        ]
    )


    df = df[
        keep
    ].copy()


    df["도로명"] = clean_text(
        df["도로명"]
    )

    df["구간명"] = clean_text(
        df["구간명"]
    )


    numeric_cols = [
        "평균속도_kmh",
        "최저속도_kmh",
        "최고속도_kmh",
        "교통량_대",
    ]


    for col in numeric_cols:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )


    # --------------------------------------------------------
    # 대표 혼잡시간 정리
    # --------------------------------------------------------

    if (
        "대표혼잡시간"
        in df.columns
    ):

        df["대표혼잡시간"] = (
            df["대표혼잡시간"]
            .astype("string")
            .str.extract(
                r"(\d{1,2})",
                expand=False,
            )
        )


        df["대표혼잡시간"] = (
            pd.to_numeric(
                df["대표혼잡시간"],
                errors="coerce",
            )
            .astype("Int64")
        )


        invalid = (
            df["대표혼잡시간"].notna()
            & ~df[
                "대표혼잡시간"
            ].between(
                0,
                23,
            )
        )


        df.loc[
            invalid,
            "대표혼잡시간",
        ] = pd.NA


    before = len(df)


    # --------------------------------------------------------
    # 결측 / 이상값 제거
    # --------------------------------------------------------

    df = df.dropna(
        subset=required
    ).copy()


    df = df[
        (
            df["평균속도_kmh"] > 0
        )
        & (
            df["교통량_대"] >= 0
        )
    ].copy()


    # --------------------------------------------------------
    # 동일 도로구간 중복 집계
    # --------------------------------------------------------

    agg = {

        "평균속도_kmh":
            "mean",

        "교통량_대":
            "mean",
    }


    if (
        "최저속도_kmh"
        in df.columns
    ):

        agg[
            "최저속도_kmh"
        ] = "min"


    if (
        "최고속도_kmh"
        in df.columns
    ):

        agg[
            "최고속도_kmh"
        ] = "max"


    if (
        "대표혼잡시간"
        in df.columns
    ):

        agg[
            "대표혼잡시간"
        ] = lambda s: (
            s.dropna()
            .mode()
            .iloc[0]
            if not s.dropna().empty
            else pd.NA
        )


    df = (
        df.groupby(
            [
                "도로명",
                "구간명",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(agg)
    )


    # --------------------------------------------------------
    # 식별자 생성
    # --------------------------------------------------------

    df.insert(
        0,
        "도로구간_id",
        (
            df["도로명"]
            + "__"
            + df["구간명"]
        )
        .str.replace(
            r"\s+",
            "_",
            regex=True,
        ),
    )


    df["원본파일"] = (
        path.name
    )


    df.to_csv(
        ROAD_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )


    print(
        "[도로 소통통계] "
        f"{before:,}행 → "
        f"{len(df):,}개 도로구간"
    )


    return df


# ============================================================
# 2. 스마트교차로
# ============================================================

def process_smart_intersection(
    path: Path | str | None = None,
):

    path = (
        Path(path)
        if path
        else find_excel_by_sheet(
            "스마트교차로_통계"
        )
    )


    df = pd.read_excel(
        path,
        sheet_name="스마트교차로_통계",
    )


    df.columns = [
        str(col).strip()
        for col in df.columns
    ]


    required = [
        "일자",
        "교차로명",
        "접근로명",
        "합계",
    ]


    hour_cols = [
        f"{hour:02d}시"
        for hour in range(24)
    ]


    missing = [
        col
        for col in (
            required
            + hour_cols
        )
        if col not in df.columns
    ]


    if missing:

        raise ValueError(
            "스마트교차로 필수 컬럼이 없습니다: "
            f"{missing}"
        )


    df = df[
        required
        + hour_cols
    ].copy()


    # --------------------------------------------------------
    # 데이터 타입 정리
    # --------------------------------------------------------

    df["일자"] = pd.to_datetime(
        df["일자"],
        errors="coerce",
    )


    df["교차로명"] = clean_text(
        df["교차로명"]
    )


    df["접근로명"] = clean_text(
        df["접근로명"]
    )


    df["합계"] = pd.to_numeric(
        df["합계"],
        errors="coerce",
    )


    for col in hour_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )


    df = df.dropna(
        subset=[
            "일자",
            "교차로명",
            "접근로명",
        ]
    ).copy()


    df[
        hour_cols
    ] = (
        df[hour_cols]
        .fillna(0)
    )


    # --------------------------------------------------------
    # 24시간 합계 검증
    # --------------------------------------------------------

    df[
        "시간대합계_계산"
    ] = (
        df[hour_cols]
        .sum(axis=1)
    )


    df[
        "합계차이"
    ] = (
        df["합계"]
        - df[
            "시간대합계_계산"
        ]
    )


    # --------------------------------------------------------
    # wide → long
    # --------------------------------------------------------

    hourly = df.melt(

        id_vars=[
            "일자",
            "교차로명",
            "접근로명",
            "합계",
            "시간대합계_계산",
            "합계차이",
        ],

        value_vars=hour_cols,

        var_name="시간대",

        value_name="교통량_대",
    )


    hourly["시간"] = (
        hourly["시간대"]
        .str.extract(
            r"(\d{2})"
        )
        .astype(int)
    )


    hourly = hourly.drop(
        columns="시간대"
    )


    hourly["교차로_id"] = (
        hourly["교차로명"]
    )


    hourly["접근로_id"] = (
        hourly["교차로명"]
        + "__"
        + hourly["접근로명"]
    )


    hourly["원본파일"] = (
        path.name
    )


    hourly = (
        hourly[
            [
                "일자",
                "시간",
                "교차로_id",
                "교차로명",
                "접근로_id",
                "접근로명",
                "교통량_대",
                "합계",
                "시간대합계_계산",
                "합계차이",
                "원본파일",
            ]
        ]
        .sort_values(
            [
                "일자",
                "교차로명",
                "접근로명",
                "시간",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # --------------------------------------------------------
    # 교차로별 일교통량
    # --------------------------------------------------------

    daily = (
        hourly
        .groupby(
            [
                "일자",
                "교차로명",
            ],
            as_index=False,
        )
        .agg(
            접근로수=(
                "접근로명",
                "nunique",
            ),
            일교통량_대=(
                "교통량_대",
                "sum",
            ),
        )
    )


    # --------------------------------------------------------
    # 첨두시간 / 첨두교통량
    # --------------------------------------------------------

    hourly_intersection = (
        hourly
        .groupby(
            [
                "일자",
                "교차로명",
                "시간",
            ],
            as_index=False,
        )[
            "교통량_대"
        ]
        .sum()
    )


    peak_idx = (
        hourly_intersection
        .groupby(
            [
                "일자",
                "교차로명",
            ]
        )[
            "교통량_대"
        ]
        .idxmax()
    )


    peak = (
        hourly_intersection
        .loc[
            peak_idx,
            [
                "일자",
                "교차로명",
                "시간",
                "교통량_대",
            ],
        ]
        .rename(
            columns={
                "시간":
                    "첨두시간",

                "교통량_대":
                    "첨두교통량_대",
            }
        )
    )


    daily = daily.merge(
        peak,
        on=[
            "일자",
            "교차로명",
        ],
        how="left",
        validate="one_to_one",
    )


    daily["원본파일"] = (
        path.name
    )


    hourly.to_csv(
        SMART_HOURLY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )


    daily.to_csv(
        SMART_DAILY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )


    mismatch = (
        df["합계차이"]
        .abs()
        > 0
    ).sum()


    print(
        "[스마트교차로] "
        f"{df['교차로명'].nunique():,}개 교차로 / "
        f"{len(df):,}개 접근로 → "
        f"{len(hourly):,}개 시간대 행"
    )


    if mismatch:

        print(
            "[검증] 원본 합계와 "
            "24시간 합계가 다른 접근로: "
            f"{mismatch:,}개"
        )


    return hourly, daily


# ============================================================
# 3. 불법주정차
# ============================================================

def find_illegal_parking_file() -> Path:
    """
    raw 폴더에서 불법주정차 CSV 자동 탐색
    """

    candidates = []

    for path in RAW_DIR.rglob(
        "*.csv"
    ):

        name = path.name.replace(
            " ",
            "",
        )

        if (
            "불법주정차" in name
            or "주정차단속" in name
        ):

            candidates.append(
                path
            )


    if not candidates:

        raise FileNotFoundError(
            "불법주정차 단속현황 CSV를 "
            "찾지 못했습니다."
        )


    if len(candidates) > 1:

        print(
            "[경고] 불법주정차 CSV가 "
            "여러 개 발견되었습니다."
        )

        for path in candidates:

            print(
                f"  - {path.name}"
            )


    return candidates[0]


def read_csv_auto(
    path: Path,
) -> pd.DataFrame:
    """
    공공데이터 CSV 인코딩 자동 처리
    """

    encodings = [
        "utf-8-sig",
        "cp949",
        "euc-kr",
        "utf-8",
    ]


    for encoding in encodings:

        try:

            df = pd.read_csv(
                path,
                encoding=encoding,
            )

            print(
                "[불법주정차] "
                f"인코딩: {encoding}"
            )

            return df

        except UnicodeDecodeError:

            continue


    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        "CSV 인코딩을 확인하지 못했습니다.",
    )


def find_column(
    columns,
    keywords,
):

    normalized = {
        col:
            re.sub(
                r"\s+",
                "",
                str(col),
            )
        for col in columns
    }


    for keyword in keywords:

        for col, clean_col in (
            normalized.items()
        ):

            if keyword in clean_col:

                return col


    return None


def process_illegal_parking(
    path: Path | str | None = None,
) -> pd.DataFrame:

    path = (
        Path(path)
        if path
        else find_illegal_parking_file()
    )


    df = read_csv_auto(
        path
    )


    df.columns = [
        str(col).strip()
        for col in df.columns
    ]


    print(
        "[불법주정차] 원본 컬럼:"
    )

    print(
        list(df.columns)
    )


    # --------------------------------------------------------
    # 컬럼 자동 탐색
    # --------------------------------------------------------

    date_col = find_column(
        df.columns,
        [
            "단속일자",
            "단속일",
            "단속날짜",
            "일자",
        ],
    )


    time_col = find_column(
        df.columns,
        [
            "단속시간",
            "시간",
        ],
    )


    dong_col = find_column(
        df.columns,
        [
            "단속동",
            "행정동",
            "읍면동",
            "동명",
        ],
    )


    place_col = find_column(
        df.columns,
        [
            "단속장소",
            "단속위치",
            "장소",
            "위치",
            "주소",
        ],
    )


    # --------------------------------------------------------
    # 최소 필수 컬럼 검증
    # --------------------------------------------------------

    if (
        dong_col is None
        and place_col is None
    ):

        raise ValueError(
            "불법주정차 데이터에서 "
            "단속동 또는 단속장소 컬럼을 "
            "찾지 못했습니다."
        )


    result = pd.DataFrame()


    # --------------------------------------------------------
    # 단속일자
    # --------------------------------------------------------

    if date_col is not None:

        result["단속일자"] = (
            pd.to_datetime(
                df[date_col],
                errors="coerce",
            )
        )

    else:

        result[
            "단속일자"
        ] = pd.NaT


    # --------------------------------------------------------
    # 단속시간
    # --------------------------------------------------------

    if time_col is not None:

        result["단속시간_원본"] = (
            df[time_col]
            .astype("string")
            .str.strip()
        )


        hour = (
            result[
                "단속시간_원본"
            ]
            .str.extract(
                r"(\d{1,2})",
                expand=False,
            )
        )


        result["단속시간"] = (
            pd.to_numeric(
                hour,
                errors="coerce",
            )
            .astype("Int64")
        )


        invalid_hour = (
            result[
                "단속시간"
            ].notna()
            & ~result[
                "단속시간"
            ].between(
                0,
                23,
            )
        )


        result.loc[
            invalid_hour,
            "단속시간",
        ] = pd.NA

    else:

        result[
            "단속시간"
        ] = pd.NA


    # --------------------------------------------------------
    # 단속동
    # --------------------------------------------------------

    if dong_col is not None:

        result["단속동_원본"] = (
            df[dong_col]
            .astype("string")
            .str.replace(
                r"\s+",
                " ",
                regex=True,
            )
            .str.strip()
        )

    else:

        result[
            "단속동_원본"
        ] = pd.NA


    # --------------------------------------------------------
    # 단속장소
    # --------------------------------------------------------

    if place_col is not None:

        result["단속장소"] = (
            df[place_col]
            .astype("string")
            .str.replace(
                r"\s+",
                " ",
                regex=True,
            )
            .str.strip()
        )

    else:

        result[
            "단속장소"
        ] = pd.NA


    # --------------------------------------------------------
    # 연/월/요일 추가
    # --------------------------------------------------------

    result["연도"] = (
        result["단속일자"]
        .dt.year
        .astype("Int64")
    )


    result["월"] = (
        result["단속일자"]
        .dt.month
        .astype("Int64")
    )


    result["요일"] = (
        result["단속일자"]
        .dt.dayofweek
        .map(
            {
                0: "월",
                1: "화",
                2: "수",
                3: "목",
                4: "금",
                5: "토",
                6: "일",
            }
        )
    )


    # --------------------------------------------------------
    # 식별자
    # --------------------------------------------------------

    result.insert(
        0,
        "단속_id",
        range(
            1,
            len(result) + 1,
        ),
    )


    result["원본파일"] = (
        path.name
    )


    # --------------------------------------------------------
    # 완전히 정보 없는 행 제거
    # --------------------------------------------------------

    empty_location = (
        result[
            "단속동_원본"
        ].isna()
        & result[
            "단속장소"
        ].isna()
    )


    result = (
        result[
            ~empty_location
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    result.to_csv(
        ILLEGAL_PARKING_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )


    print(
        "[불법주정차] "
        f"{len(df):,}행 → "
        f"{len(result):,}행"
    )


    if (
        "단속동_원본"
        in result.columns
    ):

        print(
            "- 단속동 종류: "
            f"{result['단속동_원본'].nunique(dropna=True):,}개"
        )


    return result


# ============================================================
# main
# ============================================================

def main():

    road = (
        process_road_traffic()
    )

    hourly, daily = (
        process_smart_intersection()
    )

    illegal = (
        process_illegal_parking()
    )


    print(
        "\n=============================="
    )

    print(
        "교통 데이터 전처리 완료"
    )

    print(
        "=============================="
    )


    print(
        f"- {ROAD_OUTPUT}"
    )

    print(
        f"- {SMART_HOURLY_OUTPUT}"
    )

    print(
        f"- {SMART_DAILY_OUTPUT}"
    )

    print(
        f"- {ILLEGAL_PARKING_OUTPUT}"
    )


    print(
        f"- 도로구간: "
        f"{len(road):,}개"
    )

    print(
        f"- 스마트교차로 시간대 행: "
        f"{len(hourly):,}개"
    )

    print(
        f"- 스마트교차로 일별 행: "
        f"{len(daily):,}개"
    )

    print(
        f"- 불법주정차 단속: "
        f"{len(illegal):,}건"
    )


if __name__ == "__main__":
    main()