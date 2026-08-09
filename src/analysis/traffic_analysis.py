from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 경로
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MAPPING_DIR = ROOT_DIR / "data" / "mapping"

FIGURE_DIR = (
    ROOT_DIR
    / "figures"
    / "traffic"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ROAD_INPUT = (
    PROCESSED_DIR
    / "교통_도로구간_소통통계.csv"
)

SMART_INPUT = (
    PROCESSED_DIR
    / "교통_스마트교차로_일별요약.csv"
)

PARKING_INPUT = (
    PROCESSED_DIR
    / "교통_불법주정차_행정동매핑.csv"
)

POP_INPUT = (
    PROCESSED_DIR
    / "취약인구_읍면동_기초.csv"
)


ROAD_MAPPING = (
    MAPPING_DIR
    / "도로구간_행정동매핑.csv"
)

SMART_MAPPING = (
    MAPPING_DIR
    / "스마트교차로_행정동매핑.csv"
)


OUTPUT = (
    PROCESSED_DIR
    / "교통_읍면동_지수.csv"
)


# ============================================================
# 행정동
# ============================================================

DONGNAM = [
    "목천읍", "풍세면", "광덕면", "북면",
    "성남면", "수신면", "병천면", "동면",
    "중앙동", "문성동", "원성1동", "원성2동",
    "봉명동", "일봉동", "신방동", "청룡동",
    "신안동",
]

SEOBUK = [
    "성환읍", "성거읍", "직산읍", "입장면",
    "성정1동", "성정2동",
    "쌍용1동", "쌍용2동", "쌍용3동",
    "백석동", "불당1동", "불당2동",
    "부성1동", "부성2동",
]

TARGET_DONGS = (
    DONGNAM
    + SEOBUK
)


plt.rcParams[
    "font.family"
] = "Malgun Gothic"

plt.rcParams[
    "axes.unicode_minus"
] = False


# ============================================================
# Min-Max
# ============================================================

def minmax(series):

    series = pd.to_numeric(
        series,
        errors="coerce",
    )

    minimum = series.min()
    maximum = series.max()

    if (
        pd.isna(minimum)
        or pd.isna(maximum)
    ):

        return pd.Series(
            pd.NA,
            index=series.index,
            dtype="Float64",
        )

    if maximum == minimum:

        return pd.Series(
            0.0,
            index=series.index,
        )

    return (
        series - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# 도로
# ============================================================

def aggregate_road():

    road = pd.read_csv(
        ROAD_INPUT,
        encoding="utf-8-sig",
    )

    mapping = pd.read_csv(
        ROAD_MAPPING,
        encoding="utf-8-sig",
    )


    missing = (
        ~mapping[
            "행정동"
        ].isin(
            TARGET_DONGS
        )
    ).sum()

    print(
        f"[도로] 미매핑 {missing}개"
    )


    road = road.merge(
        mapping[
            [
                "도로구간_id",
                "행정동",
            ]
        ],
        on="도로구간_id",
        how="left",
    )


    road = road[
        road[
            "행정동"
        ].isin(
            TARGET_DONGS
        )
    ]


    return (
        road
        .groupby(
            "행정동",
            as_index=False,
        )
        .agg(
            평균속도_kmh=(
                "평균속도_kmh",
                "mean",
            ),

            평균교통량_대=(
                "교통량_대",
                "mean",
            ),

            도로구간수=(
                "도로구간_id",
                "nunique",
            ),
        )
    )


# ============================================================
# 스마트교차로
# ============================================================

def aggregate_smart():

    smart = pd.read_csv(
        SMART_INPUT,
        encoding="utf-8-sig",
    )

    mapping = pd.read_csv(
        SMART_MAPPING,
        encoding="utf-8-sig",
    )


    # TMAP에서 행정동 자체가 유효하면 사용
    # 자동확정 + 검토필요 모두 활용
    mapping[
        "분석행정동"
    ] = mapping[
        "행정동"
    ].where(
        mapping[
            "행정동"
        ].isin(
            TARGET_DONGS
        )
    )


    missing = (
        mapping[
            "분석행정동"
        ]
        .isna()
        .sum()
    )


    print(
        f"[스마트교차로] "
        f"미매핑 {missing}개"
    )


    smart = smart.merge(
        mapping[
            [
                "교차로명",
                "분석행정동",
            ]
        ].rename(
            columns={
                "분석행정동":
                    "행정동"
            }
        ),
        on="교차로명",
        how="left",
    )


    smart = smart[
        smart[
            "행정동"
        ].isin(
            TARGET_DONGS
        )
    ]


    return (
        smart
        .groupby(
            "행정동",
            as_index=False,
        )
        .agg(
            평균일교통량_대=(
                "일교통량_대",
                "mean",
            ),

            평균첨두교통량_대=(
                "첨두교통량_대",
                "mean",
            ),

            교차로수=(
                "교차로명",
                "nunique",
            ),
        )
    )


# ============================================================
# 불법주정차
# ============================================================

def aggregate_parking():

    parking = pd.read_csv(
        PARKING_INPUT,
        encoding="utf-8-sig",
    )


    valid = parking[
        parking[
            "행정동"
        ].isin(
            TARGET_DONGS
        )
    ].copy()


    print(
        f"[불법주정차] "
        f"매핑 {len(valid):,}/{len(parking):,}건"
    )


    return (
        valid
        .groupby(
            "행정동",
            as_index=False,
        )
        .agg(
            불법주정차단속건수=(
                "단속_id",
                "count",
            )
        )
    )


# ============================================================
# 데이터 결합
# ============================================================

def build_dataset():

    road = aggregate_road()
    smart = aggregate_smart()
    parking = aggregate_parking()


    base = pd.DataFrame(
        {
            "행정동":
                TARGET_DONGS
        }
    )


    base[
        "구"
    ] = base[
        "행정동"
    ].apply(
        lambda x:
        "동남구"
        if x in DONGNAM
        else "서북구"
    )


    result = (
        base
        .merge(
            road,
            on="행정동",
            how="left",
        )
        .merge(
            smart,
            on="행정동",
            how="left",
        )
        .merge(
            parking,
            on="행정동",
            how="left",
        )
    )


    # --------------------------------------------------------
    # 인구
    # --------------------------------------------------------

    pop = pd.read_csv(
        POP_INPUT,
        encoding="utf-8-sig",
    )


    result = result.merge(
        pop[
            [
                "행정동",
                "총인구",
            ]
        ],
        on="행정동",
        how="left",
    )


    # 주정차 데이터가 매핑은 됐지만
    # 해당 지역 건수가 없는 경우 0
    result[
        "불법주정차단속건수"
    ] = result[
        "불법주정차단속건수"
    ].fillna(
        0
    )


    result[
        "인구천명당_불법주정차"
    ] = (
        result[
            "불법주정차단속건수"
        ]
        /
        result[
            "총인구"
        ]
        * 1000
    )


    return result


# ============================================================
# 최종 지수
# ============================================================

def calculate_index(df):

    df = df.copy()


    # --------------------------------------------------------
    # 1. 속도 위험
    # --------------------------------------------------------

    df[
        "속도위험"
    ] = (
        1
        - minmax(
            df[
                "평균속도_kmh"
            ]
        )
    )


    # --------------------------------------------------------
    # 2. 교통량 위험
    # --------------------------------------------------------

    df[
        "도로교통량_norm"
    ] = minmax(
        df[
            "평균교통량_대"
        ]
    )


    df[
        "교차로일교통량_norm"
    ] = minmax(
        df[
            "평균일교통량_대"
        ]
    )


    df[
        "교통량위험"
    ] = (
        df[
            [
                "도로교통량_norm",
                "교차로일교통량_norm",
            ]
        ]
        .mean(
            axis=1,
            skipna=True,
        )
    )


    # --------------------------------------------------------
    # 3. 첨두교통량
    # --------------------------------------------------------

    df[
        "첨두교통량위험"
    ] = minmax(
        df[
            "평균첨두교통량_대"
        ]
    )


    # --------------------------------------------------------
    # 4. 불법주정차
    # --------------------------------------------------------

    df[
        "불법주정차위험"
    ] = minmax(
        df[
            "인구천명당_불법주정차"
        ]
    )


    risk_cols = [
        "속도위험",
        "교통량위험",
        "첨두교통량위험",
        "불법주정차위험",
    ]


    df[
        "사용가능지표수"
    ] = (
        df[
            risk_cols
        ]
        .notna()
        .sum(
            axis=1
        )
    )


    # --------------------------------------------------------
    # 동일가중치 0.25
    #
    # 결측인 지표는 제외 후
    # 나머지 지표 동일가중 평균
    # --------------------------------------------------------

    df[
        "교통혼잡도"
    ] = (
        df[
            risk_cols
        ]
        .mean(
            axis=1,
            skipna=True,
        )
    )


    # 최소 2개 이상의 지표 확보
    df.loc[
        df[
            "사용가능지표수"
        ] < 2,
        "교통혼잡도",
    ] = pd.NA


    df[
        "교통혼잡도순위"
    ] = (
        df[
            "교통혼잡도"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(
            "Int64"
        )
    )


    return df


# ============================================================
# 그래프
# ============================================================

def make_figures(df):

    top10 = (
        df
        .dropna(
            subset=[
                "교통혼잡도"
            ]
        )
        .nlargest(
            10,
            "교통혼잡도",
        )
        .sort_values(
            "교통혼잡도"
        )
    )


    plt.figure(
        figsize=(9, 6)
    )

    plt.barh(
        top10[
            "행정동"
        ],
        top10[
            "교통혼잡도"
        ],
    )

    plt.title(
        "천안시 교통혼잡도 상위 10개 지역"
    )

    plt.xlabel(
        "교통혼잡도"
    )

    plt.tight_layout()


    plt.savefig(
        FIGURE_DIR
        / "교통혼잡도_상위10.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# main
# ============================================================

def main():

    result = build_dataset()

    result = calculate_index(
        result
    )


    result = result.sort_values(
        "교통혼잡도순위",
        na_position="last",
    )


    result.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )


    make_figures(
        result
    )


    print(
        "\n=============================="
    )

    print(
        "교통혼잡도 분석 완료"
    )

    print(
        "=============================="
    )


    print(
        "\n[상위 10개]"
    )


    print(
        result[
            [
                "교통혼잡도순위",
                "행정동",
                "속도위험",
                "교통량위험",
                "첨두교통량위험",
                "불법주정차위험",
                "교통혼잡도",
                "사용가능지표수",
            ]
        ]
        .dropna(
            subset=[
                "교통혼잡도"
            ]
        )
        .head(10)
        .to_string(
            index=False
        )
    )


    print(
        "\n[교통혼잡도 결측]"
    )

    print(
        result.loc[
            result[
                "교통혼잡도"
            ].isna(),
            [
                "행정동",
                "사용가능지표수",
            ],
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()