from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 경로
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

FIGURE_DIR = (
    ROOT_DIR
    / "figures"
    / "map"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TRAFFIC_INPUT = (
    PROCESSED_DIR
    / "교통_읍면동_지수.csv"
)


# ============================================================
# 천안시 31개 읍면동
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


# ============================================================
# 그래프 한글
# ============================================================

plt.rcParams[
    "font.family"
] = "Malgun Gothic"

plt.rcParams[
    "axes.unicode_minus"
] = False


# ============================================================
# SGIS 경계 파일 찾기
# ============================================================

def load_boundary():

    files = (
        list(
            RAW_DIR.rglob(
                "*.shp"
            )
        )
        +
        list(
            RAW_DIR.rglob(
                "*.geojson"
            )
        )
    )

    if not files:

        raise FileNotFoundError(
            "data/raw 안에서 "
            "SGIS shp/geojson 파일을 찾지 못했습니다."
        )

    frames = []

    for path in files:

        try:

            gdf = gpd.read_file(
                path
            )

        except Exception:

            continue


        # 행정동명 컬럼 후보
        name_candidates = [
            "adm_nm",
            "ADM_NM",
            "adm_name",
            "EMD_KOR_NM",
            "행정동",
            "행정동명",
        ]

        name_col = None

        for col in name_candidates:

            if col in gdf.columns:

                name_col = col
                break


        if name_col is None:

            continue


        temp = gdf[
            [
                name_col,
                "geometry",
            ]
        ].copy()

        temp = temp.rename(
            columns={
                name_col:
                    "행정동원본"
            }
        )

        frames.append(
            temp
        )


    if not frames:

        raise ValueError(
            "행정동명 컬럼이 있는 SGIS 경계 파일을 "
            "찾지 못했습니다."
        )


    # CRS 맞춘 뒤 결합
    base_crs = frames[0].crs

    converted = []

    for frame in frames:

        if (
            frame.crs is not None
            and base_crs is not None
            and frame.crs != base_crs
        ):

            frame = frame.to_crs(
                base_crs
            )

        converted.append(
            frame
        )


    boundary = gpd.GeoDataFrame(
        pd.concat(
            converted,
            ignore_index=True,
        ),
        crs=base_crs,
    )


    # SGIS adm_nm은
    # "충청남도 천안시 동남구 청룡동"
    # 형태일 수도 있으므로 마지막 값 추출
    boundary[
        "행정동"
    ] = (
        boundary[
            "행정동원본"
        ]
        .astype(str)
        .str.strip()
        .str.split()
        .str[-1]
    )


    boundary = boundary[
        boundary[
            "행정동"
        ].isin(
            TARGET_DONGS
        )
    ].copy()


    # 같은 행정동이 중복되어 있으면 합치기
    boundary = boundary.dissolve(
        by="행정동",
        as_index=False,
    )


    print(
        f"SGIS 경계 매칭: "
        f"{len(boundary)}/31개"
    )


    missing = (
        set(
            TARGET_DONGS
        )
        -
        set(
            boundary[
                "행정동"
            ]
        )
    )

    if missing:

        print(
            "경계 미매칭:",
            sorted(
                missing
            )
        )


    return boundary


# ============================================================
# 취약인구 결과 자동 찾기
# ============================================================

def load_vulnerability():

    files = list(
        PROCESSED_DIR.glob(
            "*취약인구*.csv"
        )
    )

    score_candidates = [
        "취약인구지수",
        "취약인구지수_V1",
        "취약인구_V1",
        "V1_취약인구지수",
        "V1",
    ]

    for path in files:

        try:

            df = pd.read_csv(
                path,
                encoding="utf-8-sig",
            )

        except Exception:

            continue


        if "행정동" not in df.columns:
            continue


        for col in score_candidates:

            if col in df.columns:

                print(
                    f"취약인구 파일: {path.name}"
                )

                print(
                    f"취약인구 컬럼: {col}"
                )

                return (
                    df[
                        [
                            "행정동",
                            col,
                        ]
                    ]
                    .rename(
                        columns={
                            col:
                                "취약인구지수"
                        }
                    )
                )


    raise FileNotFoundError(
        "취약인구 V1 결과 파일을 "
        "자동으로 찾지 못했습니다."
    )


# ============================================================
# 지도 그리기
# ============================================================

def draw_map(
    boundary,
    data,
    value_col,
    title,
    output_name,
):

    gdf = boundary.merge(
        data,
        on="행정동",
        how="left",
    )


    fig, ax = plt.subplots(
        figsize=(9, 10)
    )


    # 값이 있는 지역
    gdf.plot(
        column=value_col,
        ax=ax,
        legend=True,
        edgecolor="black",
        linewidth=0.5,
        missing_kwds={
            "color": "lightgrey",
            "label": "데이터 없음",
        },
        legend_kwds={
            "label": value_col,
            "shrink": 0.7,
        },
    )


    # 행정동명 표시
    label_gdf = gdf.copy()

    label_gdf[
        "label_point"
    ] = label_gdf.geometry.representative_point()


    for _, row in label_gdf.iterrows():

        point = row[
            "label_point"
        ]

        ax.text(
            point.x,
            point.y,
            row["행정동"],
            ha="center",
            va="center",
            fontsize=7,
        )


    ax.set_title(
        title,
        fontsize=15,
        pad=15,
    )

    ax.axis(
        "off"
    )


    plt.tight_layout()


    output = (
        FIGURE_DIR
        / output_name
    )

    plt.savefig(
        output,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()


    print(
        f"생성: {output}"
    )


# ============================================================
# main
# ============================================================

def main():

    boundary = (
        load_boundary()
    )


    # --------------------------------------------------------
    # 취약인구
    # --------------------------------------------------------

    vulnerability = (
        load_vulnerability()
    )

    draw_map(
        boundary=boundary,
        data=vulnerability,
        value_col="취약인구지수",
        title="천안시 읍면동별 취약인구 지수",
        output_name="취약인구지수_지도.png",
    )


    # --------------------------------------------------------
    # 교통혼잡도
    # --------------------------------------------------------

    traffic = pd.read_csv(
        TRAFFIC_INPUT,
        encoding="utf-8-sig",
    )


    draw_map(
        boundary=boundary,
        data=traffic[
            [
                "행정동",
                "교통혼잡도",
            ]
        ],
        value_col="교통혼잡도",
        title="천안시 읍면동별 교통혼잡도",
        output_name="교통혼잡도_지도.png",
    )


    print(
        "\n지도 시각화 완료"
    )


if __name__ == "__main__":
    main()